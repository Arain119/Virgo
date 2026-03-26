"""Hyperparameter optimization entrypoint (Optuna) for PPO training.

This module runs an Optuna study to tune PPO hyperparameters and persists:
- The best parameters to `BEST_PARAMS_PATH`
- A progress JSON file to `PROGRESS_FILE` for UI progress monitoring

It is typically invoked as a separate process (from the desktop UI or automation):
`python -m virgo_trader.training.optimize ...`
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import signal
from pathlib import Path
from typing import Any

STUDY_NAME = "ppo-stock-trading-study"
N_STARTUP_TRIALS = 5  # For Optuna MedianPruner

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PACKAGE_DIR / "configs" / "hyperparameters.json"

try:
    _CONFIG_DATA = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    _CONFIG_DATA = {}

ARCHITECTURE_CONFIG = _CONFIG_DATA.get("model_architecture", {})
CNN_ARCH_CONFIG = ARCHITECTURE_CONFIG.get("cnn", {})
TRANSFORMER_ARCH_CONFIG = ARCHITECTURE_CONFIG.get("transformer", {})
CROSS_ATTENTION_ARCH_CONFIG = ARCHITECTURE_CONFIG.get("cross_attention", {})
DEFAULT_AGENT_TYPE = ARCHITECTURE_CONFIG.get("default_agent_type", "multiscale_cnn")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials.")
    parser.add_argument(
        "--timeout", type=int, default=120, help="Timeout for optimization (minutes)."
    )
    parser.add_argument(
        "--stock_code", type=str, default="510050.SH", help="Stock code for trading."
    )
    parser.add_argument("--start_date", type=str, default="20210101", help="Start date for data.")
    parser.add_argument("--end_date", type=str, default="20231231", help="End date for data.")
    parser.add_argument(
        "--n_splits", type=int, default=4, help="Number of splits for walk-forward validation."
    )
    parser.add_argument(
        "--window_size_min",
        type=int,
        default=10,
        help="Minimum window size for optimization search.",
    )
    parser.add_argument(
        "--window_size_max",
        type=int,
        default=60,
        help="Maximum window size for optimization search.",
    )
    parser.add_argument(
        "--multiplier_min",
        type=int,
        default=5,
        help="Minimum training multiplier for optimization search.",
    )
    parser.add_argument(
        "--multiplier_max",
        type=int,
        default=20,
        help="Maximum training multiplier for optimization search.",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        choices=["multiscale_cnn", "transformer", "cross_attention"],
        default=DEFAULT_AGENT_TYPE,
        help="Policy backbone to optimize.",
    )
    parser.add_argument(
        "--sentiment_dataset", type=str, default="", help="Optional sentiment JSONL dataset path."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows multiprocessing support (no-op on other platforms).
    multiprocessing.freeze_support()

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Configure environment before importing optional ML stacks (SB3, TensorBoard, TensorFlow).
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # Heavy imports are intentionally delayed so `--help` works without ML deps installed.
    import numpy as np
    import optuna
    import pandas as pd
    import psutil
    import torch
    from optuna.pruners import MedianPruner
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from virgo_trader.agent.policy_network import (
        CNNActorCriticPolicy,
        CrossAttentionActorCriticPolicy,
        TransformerActorCriticPolicy,
    )
    from virgo_trader.data.data_fetcher import get_stock_data
    from virgo_trader.data.dataset_builder import attach_sentiment_features, load_sentiment_loader
    from virgo_trader.data.feature_engineer import FeatureEngineer
    from virgo_trader.environment.trading_env import TradingEnv
    from virgo_trader.utils.paths import (
        BEST_PARAMS_PATH,
        PROGRESS_FILE,
        STUDY_DB_PATH,
        ensure_dir,
        migrate_legacy_files,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    migrate_legacy_files()
    ensure_dir(PROGRESS_FILE.parent)

    if args.seed is not None:
        set_random_seed(args.seed)

    sentiment_loader, sentiment_columns = load_sentiment_loader(
        args.sentiment_dataset.strip() or None
    )
    args.sentiment_loader = sentiment_loader
    args.sentiment_columns = sentiment_columns
    if sentiment_loader is not None and sentiment_columns:
        logging.info(
            "Loaded sentiment dataset %s (features=%d) for optimization.",
            args.sentiment_dataset,
            len(sentiment_columns),
        )

    class GracefulShutdownCallback(BaseCallback):
        """Callback that stops training when a shutdown event is set."""

        def __init__(self, shutdown_event: multiprocessing.Event, verbose: int = 0):
            super().__init__(verbose)
            self.shutdown_event = shutdown_event

        def _on_step(self) -> bool:
            if self.shutdown_event.is_set():
                if self.verbose > 0:
                    print("Shutdown signal received, stopping training...")
                return False
            return True

    def save_progress_callback(study: Any, trial: Any) -> None:
        """Persist progress after each Optuna trial for UI monitoring."""
        try:
            progress_data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            progress_data = {"trials": [], "best_trial": None}

        trial_info = {"number": trial.number, "value": trial.value, "params": trial.params}
        progress_data["trials"].append(trial_info)

        try:
            best_trial = study.best_trial
        except ValueError:
            best_trial = None

        if best_trial is not None:
            stored_best = progress_data.get("best_trial")
            if stored_best is None or best_trial.value > stored_best.get("value", float("-inf")):
                progress_data["best_trial"] = {
                    "number": best_trial.number,
                    "value": best_trial.value,
                    "params": best_trial.params,
                }

        PROGRESS_FILE.write_text(json.dumps(progress_data, indent=4), encoding="utf-8")

    def objective(
        trial: Any,
        raw_data: pd.DataFrame,
        shutdown_event: multiprocessing.Event,
    ) -> float:
        """Optuna objective for PPO hyperparameter tuning."""

        agent_params = {
            "learning_rate": trial.suggest_float("lr", 5e-6, 5e-4, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
            "n_epochs": trial.suggest_int("n_epochs", 5, 25),
            "n_steps": trial.suggest_int("n_steps", 512, 4096, step=128),
            "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.8),
            "ent_coef": trial.suggest_float("ent_coef", 0.005, 0.02),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        }

        window_size = trial.suggest_int("window_size", args.window_size_min, args.window_size_max)
        train_multiplier = trial.suggest_int(
            "train_multiplier", args.multiplier_min, args.multiplier_max
        )

        agent_type = (getattr(args, "agent_type", DEFAULT_AGENT_TYPE) or DEFAULT_AGENT_TYPE).lower()
        if agent_type == "transformer":
            policy_class = TransformerActorCriticPolicy
            policy_kwargs = {
                "features_extractor_kwargs": {
                    "features_dim": TRANSFORMER_ARCH_CONFIG.get("features_dim", 128),
                    "d_model": TRANSFORMER_ARCH_CONFIG.get("d_model", 128),
                    "n_heads": TRANSFORMER_ARCH_CONFIG.get("n_heads", 4),
                    "depth": TRANSFORMER_ARCH_CONFIG.get("num_layers", 2),
                    "ff_dim": TRANSFORMER_ARCH_CONFIG.get("ff_dim"),
                    "dropout": TRANSFORMER_ARCH_CONFIG.get("dropout_rate", 0.1),
                    "pooling": TRANSFORMER_ARCH_CONFIG.get("pooling", "cls"),
                    "positional_encoding": TRANSFORMER_ARCH_CONFIG.get(
                        "positional_encoding", "learned"
                    ),
                    "use_relative_position_bias": TRANSFORMER_ARCH_CONFIG.get(
                        "use_relative_position_bias", True
                    ),
                    "relative_attention_buckets": TRANSFORMER_ARCH_CONFIG.get(
                        "relative_attention_buckets", 32
                    ),
                    "relative_attention_max_distance": TRANSFORMER_ARCH_CONFIG.get(
                        "relative_attention_max_distance", 256
                    ),
                }
            }
        elif agent_type == "cross_attention":
            policy_class = CrossAttentionActorCriticPolicy
            transformer_kwargs = CROSS_ATTENTION_ARCH_CONFIG.get("transformer_kwargs", {})
            policy_kwargs = {
                "features_extractor_kwargs": {
                    "features_dim": CROSS_ATTENTION_ARCH_CONFIG.get("features_dim", 128),
                    "cnn_dim": CROSS_ATTENTION_ARCH_CONFIG.get("cnn_dim", 128),
                    "transformer_dim": CROSS_ATTENTION_ARCH_CONFIG.get("transformer_dim", 128),
                    "cross_dim": CROSS_ATTENTION_ARCH_CONFIG.get("cross_dim", 128),
                    "cross_heads": CROSS_ATTENTION_ARCH_CONFIG.get("cross_heads", 4),
                    "transformer_kwargs": {
                        "d_model": transformer_kwargs.get(
                            "d_model", CROSS_ATTENTION_ARCH_CONFIG.get("transformer_dim", 128)
                        ),
                        "n_heads": transformer_kwargs.get(
                            "n_heads", CROSS_ATTENTION_ARCH_CONFIG.get("cross_heads", 4)
                        ),
                        "depth": transformer_kwargs.get("depth", 2),
                        "ff_dim": transformer_kwargs.get("ff_dim"),
                        "dropout": transformer_kwargs.get("dropout", 0.1),
                        "pooling": transformer_kwargs.get("pooling", "cls"),
                        "positional_encoding": transformer_kwargs.get(
                            "positional_encoding", "learned"
                        ),
                        "use_relative_position_bias": transformer_kwargs.get(
                            "use_relative_position_bias", True
                        ),
                        "relative_attention_buckets": transformer_kwargs.get(
                            "relative_attention_buckets", 32
                        ),
                        "relative_attention_max_distance": transformer_kwargs.get(
                            "relative_attention_max_distance", 256
                        ),
                    },
                }
            }
        else:
            policy_class = CNNActorCriticPolicy
            policy_kwargs = {
                "features_extractor_kwargs": {
                    "features_dim": CNN_ARCH_CONFIG.get("features_dim", 128)
                }
            }

        try:
            # --- Data preprocessing ---
            if raw_data is None or raw_data.empty:
                raise ValueError("raw_data is empty")

            feature_engineer = FeatureEngineer(window_size=window_size)
            processed_data = feature_engineer.process(raw_data)
            if processed_data is None or processed_data.empty:
                raise ValueError("Processed data is empty")

            # --- Sentiment features (optional) ---
            if getattr(args, "sentiment_loader", None) is not None and getattr(
                args, "sentiment_columns", None
            ):
                processed_data = attach_sentiment_features(
                    processed_df=processed_data,
                    symbol=str(args.stock_code),
                    sentiment_loader=args.sentiment_loader,
                    sentiment_columns=args.sentiment_columns,
                )

            # --- Walk-forward validation ---
            total_len = len(processed_data)
            fold_size = total_len // int(args.n_splits)
            validation_scores: list[float] = []

            for split_idx in range(fold_size, total_len, fold_size):
                if shutdown_event.is_set():
                    raise optuna.exceptions.TrialPruned()

                train_df = processed_data.iloc[:split_idx]

                # Environment creation
                train_env = make_vec_env(
                    lambda end_step=split_idx: TradingEnv(
                        stock_code=args.stock_code,
                        raw_df=raw_data,
                        window_size=window_size,
                        end_step=end_step,
                    ),
                    n_envs=1,
                    vec_env_cls=SubprocVecEnv,
                )

                shutdown_callback = GracefulShutdownCallback(shutdown_event=shutdown_event)

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA device not available for optimization stage.")
                device = "cuda"
                model = PPO(
                    policy_class,
                    train_env,
                    verbose=0,
                    device=device,
                    policy_kwargs=policy_kwargs,
                    seed=args.seed,
                    **agent_params,
                )

                train_timesteps = len(train_df) * train_multiplier
                model.learn(total_timesteps=train_timesteps, callback=shutdown_callback)

                # Validation run
                val_env = TradingEnv(
                    stock_code=args.stock_code,
                    raw_df=raw_data,
                    window_size=window_size,
                    start_step=split_idx,
                )

                obs, _ = val_env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = val_env.step(action)
                    done = bool(terminated) or bool(truncated)
                    if val_env.current_step >= split_idx + fold_size:
                        break

                performance_metrics = val_env.calculate_performance_metrics()
                sharpe = float(performance_metrics.get("sharpe_ratio", 0.0) or 0.0)
                excess = float(performance_metrics.get("excess_return", 0.0) or 0.0)
                beta_penalty = max(0.0, float(performance_metrics.get("beta", 0.0) or 0.0) - 0.5)
                corr_penalty = max(0.0, float(performance_metrics.get("correlation", 0.0) or 0.0))
                tracking_bonus = float(performance_metrics.get("tracking_error", 0.0) or 0.0) * 0.05
                score = (
                    sharpe + 0.5 * excess + tracking_bonus - 0.3 * beta_penalty - 0.2 * corr_penalty
                )
                validation_scores.append(score)

            mean_score = np.mean(validation_scores)
            return float(np.nan_to_num(mean_score, nan=-1e9, posinf=-1e9, neginf=-1e9))

        except optuna.exceptions.TrialPruned:
            logging.warning("Trial %s was pruned due to a shutdown request.", trial.number)
            return -1e9
        except Exception as exc:
            logging.error("Trial %s failed with error: %s", trial.number, exc)
            return -1e9

    # --- Study / progress setup ---
    storage_name = f"sqlite:///{STUDY_DB_PATH.as_posix()}"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=N_STARTUP_TRIALS),
    )

    PROGRESS_FILE.write_text(
        json.dumps({"trials": [], "best_trial": None}, indent=4), encoding="utf-8"
    )

    shutdown_event = multiprocessing.Event()

    def signal_handler(signum, frame) -> None:
        _ = signum, frame
        logging.info("Shutdown signal received. Setting event to stop gracefully.")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.info("Fetching and caching data before optimization starts...")
    raw_data = get_stock_data(args.stock_code, args.start_date, args.end_date)
    if raw_data is None or raw_data.empty:
        logging.error("Failed to fetch initial data. Aborting optimization.")
        return 1

    n_trials = int(args.n_trials)
    timeout_seconds = int(args.timeout) * 60

    logging.info(
        "Starting optimization with %s trials and a timeout of %s seconds.",
        n_trials,
        timeout_seconds,
    )
    main_process = psutil.Process(os.getpid())
    try:
        study.optimize(
            lambda trial: objective(trial, raw_data, shutdown_event),
            n_trials=n_trials,
            timeout=timeout_seconds,
            callbacks=[save_progress_callback],
        )
    except KeyboardInterrupt:
        logging.info("Optimization stopped by user (KeyboardInterrupt).")
    except Exception as exc:
        logging.error("An unexpected error occurred during optimization: %s", exc)
    finally:
        logging.info("Optimization finished. Cleaning up child processes...")
        children = main_process.children(recursive=True)
        for child in children:
            try:
                logging.info("Terminating child process: %s", child.pid)
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, still_alive = psutil.wait_procs(children, timeout=3)
        for proc in still_alive:
            logging.warning("Process %s did not terminate gracefully, killing it.", proc.pid)
            proc.kill()
        logging.info("Cleanup complete.")

    logging.info("Best trial value: %s", study.best_value)
    logging.info("Best parameters found:")
    for key, value in study.best_params.items():
        logging.info("  %s: %s", key, value)

    best_params_with_agent = dict(study.best_params)
    best_params_with_agent["agent_type"] = str(args.agent_type).lower()
    BEST_PARAMS_PATH.write_text(json.dumps(best_params_with_agent, indent=4), encoding="utf-8")
    logging.info("Best parameters saved to %s", BEST_PARAMS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
