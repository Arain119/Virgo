"""Training pipeline for PPO-based trading agents.

Builds environments, configures policies/callbacks (dashboard updates, early
stopping), runs training, and persists training session data to the database.
"""

import argparse
import datetime
import json
import logging
import os
import random
import shutil
import time
from email.utils import parseaddr
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
except Exception as exc:  # pragma: no cover - optional at import time for CLI help
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception as exc:  # pragma: no cover - optional at import time for CLI help
    PPO = None  # type: ignore[assignment]
    BaseCallback = object  # type: ignore[misc,assignment]
    CallbackList = object  # type: ignore[misc,assignment]
    make_vec_env = None  # type: ignore[assignment]
    DummyVecEnv = object  # type: ignore[misc,assignment]
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None

# Policy networks are imported lazily inside `train()` so `--help` works even when
# optional ML dependencies (torch/SB3) are not available in the current environment.
from ..data.data_fetcher import get_stock_data
from ..data.dataset_builder import attach_sentiment_features, load_sentiment_loader
from ..data.feature_cache import (
    feature_cache_path,
    load_cached_features_for_range,
    save_cached_features,
)
from ..data.feature_engineer import FeatureEngineer
from ..data.stock_pools import get_stock_pool_codes, get_stock_pool_label
from ..environment.multi_asset_env import MultiAssetTradingEnv
from ..environment.reward import RewardConfig
from ..environment.trading_env import TradingEnv
from ..utils.database_manager import create_training_session, init_db, save_episode_data
from ..utils.early_stopping import (
    EarlyStopConfig,
    ValidationEarlyStopCallback,
    split_datasets_by_ratio,
)
from ..utils.email_notifier import send_notification
from ..utils.paths import (
    BEST_PARAMS_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    TENSORBOARD_DIR,
    ensure_dir,
    find_existing_best_params_path,
)

CONFIG_DIR = PROJECT_ROOT / "config"
USER_CONFIG_PATH = CONFIG_DIR / "user_config.json"


class DashboardCallback(BaseCallback):
    """
    一个用于更新UI仪表盘和进度的Stable-Baselines3回调函数。
    它只发送增量更新信号。
    """

    def __init__(
        self,
        dashboard,
        progress_callback,
        total_steps,
        total_episodes,
        session_id,
        stop_flag_func=None,
        emit_every_steps: int = 10,
        emit_interval_s: float = 0.25,
        verbose=0,
    ):
        super(DashboardCallback, self).__init__(verbose)
        self.dashboard = dashboard
        self.progress_callback = progress_callback
        self.total_steps = total_steps
        self.total_episodes = total_episodes
        self.session_id = session_id
        self.stop_flag_func = stop_flag_func
        self.episode_num = 1
        self.current_episode_trades = []
        self.current_portfolio_history = []
        self._emit_every_steps = max(1, int(emit_every_steps))
        self._emit_interval_s = max(0.05, float(emit_interval_s))
        self._last_emit_time = 0.0
        self._last_emit_step = 0
        self._last_trade_count = 0
        self._trades_df_cache = pd.DataFrame()

    def _on_step(self) -> bool:
        if self.stop_flag_func and self.stop_flag_func():
            return False

        env = self.training_env.envs[0].unwrapped

        self.current_portfolio_history.append(env.total_worth)
        trade_count = len(env.trades) if env.trades else 0
        new_trades = []
        if trade_count > self._last_trade_count and env.trades:
            try:
                new_trades = list(env.trades[self._last_trade_count : trade_count])
            except Exception:
                new_trades = []
        if trade_count != self._last_trade_count:
            self.current_episode_trades = list(env.trades) if env.trades else []
            if self.current_episode_trades:
                df = pd.DataFrame(self.current_episode_trades)
                if not df.empty and "timestamp" in df.columns:
                    df = df.set_index("timestamp")
            else:
                df = pd.DataFrame()
            self._trades_df_cache = df
            self._last_trade_count = trade_count

        done = bool(self.locals["dones"][0])
        now = time.monotonic()
        should_emit = done or (
            (self.n_calls - self._last_emit_step) >= self._emit_every_steps
            and (now - self._last_emit_time) >= self._emit_interval_s
        )

        if should_emit and self.progress_callback:
            serialized_trades = []
            for trade in new_trades:
                if not isinstance(trade, dict):
                    continue
                payload = dict(trade)
                ts = payload.get("timestamp")
                if ts is not None:
                    try:
                        payload["timestamp"] = pd.Timestamp(ts).isoformat()
                    except Exception:
                        payload["timestamp"] = str(ts)
                for key in ("price", "quantity"):
                    if key in payload:
                        try:
                            payload[key] = float(payload[key])
                        except (TypeError, ValueError):
                            pass
                serialized_trades.append(payload)

            update_data = {
                "total_worth": float(env.total_worth),
                "portfolio_point": float(env.total_worth),
                "portfolio_index": len(self.current_portfolio_history) - 1,
                "new_trades": serialized_trades,
                "done": done,
                "current_episode": self.episode_num,
                "total_episodes": self.total_episodes,
                "symbol": getattr(env, "symbol", None),
                "trade_count": trade_count,
            }
            self.progress_callback.chart_update.emit(update_data)
            self._last_emit_time = now
            self._last_emit_step = self.n_calls

        # 更新进度条和状态
        if done:
            # 回合结束，保存完整数据到数据库
            save_episode_data(
                self.session_id,
                self.episode_num,
                self.current_portfolio_history,
                self._trades_df_cache,
            )

            # 为下一回合重置
            self.episode_num += 1
            self.current_episode_trades = []
            self.current_portfolio_history = []
            self._trades_df_cache = pd.DataFrame()
            self._last_trade_count = 0

        if self.progress_callback:
            current_steps = int(getattr(self, "num_timesteps", self.n_calls))
            progress_percent = int(100 * current_steps / max(1, int(self.total_steps)))
            self.progress_callback.progress.emit(progress_percent)
            status_msg = (
                f"训练中... 回合: {self.episode_num}, 总步数: {current_steps}/{self.total_steps}"
            )
            self.progress_callback.status.emit(status_msg)

        return True


class _SB3ValidationEarlyStopCallback(BaseCallback):
    def __init__(self, impl: ValidationEarlyStopCallback, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.impl = impl

    def _on_step(self) -> bool:
        if self.impl.should_eval(self.num_timesteps):
            return self.impl.on_eval(self.model, self.num_timesteps)
        return True


def train(train_params, dashboard=None, progress_callback=None, stop_flag_func=None):
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if _SB3_IMPORT_ERROR is not None or PPO is None or make_vec_env is None:
        message = "Training requires stable-baselines3 and a working PyTorch installation."
        raise RuntimeError(message) from _SB3_IMPORT_ERROR
    if torch is None:
        message = "Training requires a working PyTorch installation."
        raise RuntimeError(message) from _TORCH_IMPORT_ERROR
    from ..agent.policy_network import (  # local import: optional torch dependency
        CNNActorCriticPolicy,
        CrossAttentionActorCriticPolicy,
        TransformerActorCriticPolicy,
    )

    # Ensure the SQLite schema exists before recording training sessions.
    init_db()

    seed_value: Optional[int] = None
    seed_raw = train_params.get("seed")
    try:
        if seed_raw is not None and str(seed_raw).strip() != "":
            seed_value = int(seed_raw)
    except (TypeError, ValueError):
        seed_value = None

    if seed_value is not None:
        try:
            from stable_baselines3.common.utils import set_random_seed

            set_random_seed(seed_value)
        except Exception:
            np.random.seed(seed_value)
            random.seed(seed_value)
            torch.manual_seed(seed_value)
        train_params["seed"] = seed_value
    # --- 自动设备选择 ---
    # 为本次训练添加时间戳
    train_params["start_time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- 加载配置 ---
    # 构建配置文件的绝对路径，以确保在任何工作目录下都能正确找到文件
    config_path = Path(__file__).resolve().parents[1] / "configs" / "hyperparameters.json"
    default_params = json.loads(config_path.read_text(encoding="utf-8"))

    agent_params = default_params["ppo_agent"]
    reward_payload = train_params.get("reward_config")
    if isinstance(reward_payload, str):
        try:
            reward_payload = json.loads(reward_payload)
        except json.JSONDecodeError:
            reward_payload = None
    if not isinstance(reward_payload, dict):
        reward_payload = (
            default_params.get("reward") if isinstance(default_params.get("reward"), dict) else None
        )
    try:
        reward_config = RewardConfig.from_dict(reward_payload)
    except Exception:
        reward_config = RewardConfig()
    train_params["reward_config"] = reward_config.to_dict()
    architecture_config = default_params.get("model_architecture", {})
    default_agent_type = architecture_config.get("default_agent_type", "multiscale_cnn")
    cnn_arch_config = architecture_config.get("cnn", {})
    transformer_arch_config = architecture_config.get("transformer", {})
    cross_attention_arch_config = architecture_config.get("cross_attention", {})

    # 允许来自UI/命令参数的指定覆盖配置默认值
    requested_agent_type = train_params.get("agent_type", default_agent_type)
    selected_agent_type = (requested_agent_type or default_agent_type).lower()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device} for training.")

    # 尝试加载最佳参数
    best_params_path = find_existing_best_params_path() or BEST_PARAMS_PATH
    if best_params_path.exists():
        try:
            with best_params_path.open("r", encoding="utf-8") as f:
                best_params = json.load(f)

            logging.info(
                "Loaded best parameters from best_params.json. Overriding default hyperparameters."
            )
            # 应用最佳参数
            agent_params["lr"] = best_params.get("lr", agent_params["lr"])
            agent_params["gamma"] = best_params.get("gamma", agent_params["gamma"])
            agent_params["eps_clip"] = best_params.get("clip_range", agent_params["eps_clip"])
            agent_params["K_epochs"] = best_params.get("n_epochs", agent_params["K_epochs"])
            agent_params["vf_coef"] = best_params.get("vf_coef", agent_params["vf_coef"])
            agent_params["ent_coef"] = best_params.get("ent_coef", agent_params["ent_coef"])
            agent_params["max_grad_norm"] = best_params.get(
                "max_grad_norm", agent_params["max_grad_norm"]
            )
            agent_params["gae_lambda"] = best_params.get("gae_lambda", agent_params["gae_lambda"])
            agent_params["batch_size"] = best_params.get(
                "batch_size", agent_params.get("batch_size", 64)
            )
            agent_params["target_kl"] = best_params.get("target_kl", agent_params.get("target_kl"))
            train_params["learning_frequency"] = best_params.get(
                "n_steps", train_params["learning_frequency"]
            )  # n_steps 对应 learning_frequency
            train_params["window_size"] = best_params.get(
                "window_size", train_params["window_size"]
            )  # window_size

            best_agent_type = best_params.get("agent_type")
            if train_params.get("agent_type"):
                if best_agent_type and best_agent_type.lower() != selected_agent_type:
                    logging.info(
                        "best_params.json was generated for agent_type '%s', but UI selected '%s'. Proceeding with UI selection.",
                        best_agent_type,
                        selected_agent_type,
                    )
            elif best_agent_type:
                selected_agent_type = best_agent_type.lower()

        except Exception as e:
            logging.warning(f"Failed to load best_params.json: {e}. Using default hyperparameters.")
    else:
        logging.info("best_params.json not found. Using default hyperparameters.")

    training_defaults = default_params.get("training", {})
    if train_params.get("learning_frequency", 0) <= 0:
        train_params["learning_frequency"] = training_defaults.get("update_timestep", 2048)
    if train_params.get("window_size", 0) <= 0:
        train_params["window_size"] = training_defaults.get("window_size", 30)

    supported_agents = {"multiscale_cnn", "transformer", "cross_attention"}
    normalized_agent_type = (
        selected_agent_type if selected_agent_type in supported_agents else "multiscale_cnn"
    )
    train_params["agent_type"] = normalized_agent_type

    stock_codes = list(dict.fromkeys(train_params.get("stock_pool") or []))
    if not stock_codes:
        error_msg = "stock_pool 不能为空。"
        logging.error(error_msg)
        if progress_callback:
            progress_callback.status.emit(f"错误: {error_msg}")
        return
    pool_key = train_params.get("stock_pool_key")
    pool_label = train_params.get("stock_pool_label")
    if pool_key and not pool_label:
        try:
            pool_label = get_stock_pool_label(pool_key)
            train_params["stock_pool_label"] = pool_label
        except Exception:
            pool_label = pool_key
    if pool_label:
        logging.info("Using stock pool '%s' (%d symbols).", pool_label, len(stock_codes))

    start_date = train_params["start_date"]
    end_date = train_params["end_date"]
    window_size = train_params["window_size"]  # 确保这里使用更新后的window_size

    # --- 准备数据 ---
    use_calendar_features = normalized_agent_type != "multiscale_cnn"
    feature_engineer = FeatureEngineer(use_calendar_features=use_calendar_features)
    sentiment_dataset_path = train_params.get("sentiment_dataset")
    sentiment_loader, sentiment_columns = load_sentiment_loader(sentiment_dataset_path)
    if sentiment_loader and sentiment_columns:
        logging.info(
            "Loaded sentiment dataset %s (features=%d).",
            sentiment_dataset_path,
            len(sentiment_columns),
        )

    datasets = []

    for code in stock_codes:
        try:
            logging.info(f"Loading and processing data for {code}...")
            raw_df = get_stock_data(code, start_date, end_date)
            if raw_df.empty:
                logging.warning("无法获取 %s 的数据，将跳过该标的。", code)
                continue
            cache_path = feature_cache_path(
                symbol=code,
                start_date=start_date,
                end_date=end_date,
                use_calendar_features=use_calendar_features,
            )
            processed_df = load_cached_features_for_range(
                symbol=code,
                start_date=start_date,
                end_date=end_date,
                use_calendar_features=use_calendar_features,
            )
            if processed_df is None:
                processed_df = feature_engineer.process(raw_df.copy())
                save_cached_features(cache_path, processed_df)
            else:
                logging.info("Loaded feature cache for %s: %s", code, cache_path)
            processed_df = attach_sentiment_features(
                processed_df=processed_df,
                symbol=code,
                sentiment_loader=sentiment_loader,
                sentiment_columns=sentiment_columns,
            )
            datasets.append({"symbol": code, "raw_df": raw_df, "df": processed_df})
        except Exception as exc:
            logging.warning("处理 %s 数据失败：%s，已跳过。", code, exc)

    if not datasets:
        error_msg = f"无法获取任何股票数据，请检查股票代码或网络连接。输入: {stock_codes}"
        logging.error(error_msg)
        if progress_callback:
            progress_callback.status.emit(f"错误: {error_msg}")
        return

    resolved_stock_pool = [item.get("symbol") for item in datasets if item.get("symbol")]
    train_params["resolved_stock_pool"] = resolved_stock_pool
    skipped_stock_pool = [code for code in stock_codes if code not in resolved_stock_pool]
    if skipped_stock_pool:
        train_params["skipped_stock_pool"] = skipped_stock_pool
        logging.warning(
            "Some symbols were skipped due to missing/invalid data (%d/%d): %s",
            len(skipped_stock_pool),
            len(stock_codes),
            ",".join(skipped_stock_pool[:10]) + ("..." if len(skipped_stock_pool) > 10 else ""),
        )

    # --- 可选：时间序列验证集 + 早停（防止欠训练/过训练）---
    early_stop_cfg = EarlyStopConfig.from_payload(train_params.get("early_stop"))
    train_datasets = datasets
    val_datasets = []
    val_env = None
    if early_stop_cfg.enabled:
        try:
            train_datasets, val_datasets, split_info = split_datasets_by_ratio(
                datasets,
                window_size=window_size,
                split_ratio=early_stop_cfg.val_split_ratio,
            )
            train_params["validation_split"] = split_info
            logging.info(
                "Early-stop enabled (metric=%s, split=%.2f): train_rows=%s val_rows=%s",
                early_stop_cfg.metric,
                float(early_stop_cfg.val_split_ratio),
                split_info.get("train_rows"),
                split_info.get("val_rows"),
            )
        except Exception as exc:
            logging.warning("Early-stop disabled due to split failure: %s", exc)
            early_stop_cfg.enabled = False

    # --- Vectorized training envs (reduce PPO variance) ---
    n_envs_raw = train_params.get("n_envs", 1)
    try:
        n_envs = int(n_envs_raw)
    except (TypeError, ValueError):
        n_envs = 1
    n_envs = max(1, n_envs)
    train_params["n_envs"] = n_envs

    # --- Resolve effective sequence length (pool alignment) ---
    if len(train_datasets) == 1:
        dataset = train_datasets[0]
        common_index = dataset["df"].index.intersection(dataset["raw_df"].index)
        primary_sequence_length = len(common_index)
    else:
        common_index = None
        for item in train_datasets:
            df_index = item["df"].index.intersection(item["raw_df"].index)
            common_index = df_index if common_index is None else common_index.intersection(df_index)
        primary_sequence_length = len(common_index) if common_index is not None else 0
        logging.info(
            "Stock pool mode enabled: %d symbols processed simultaneously.", len(train_datasets)
        )

    # --- Random subsequence training (random start + constant horizon) ---
    subseq_payload = train_params.get("random_subsequence")
    if isinstance(subseq_payload, str):
        try:
            subseq_payload = json.loads(subseq_payload)
        except json.JSONDecodeError:
            subseq_payload = None
    if not isinstance(subseq_payload, dict):
        subseq_payload = {}

    disable_subseq = bool(train_params.get("disable_random_subsequence"))
    top_episode_length = (
        train_params.get("episode_length") or train_params.get("subsequence_length") or 0
    )
    top_random_start = train_params.get("random_start")
    try:
        top_episode_length = int(top_episode_length)
    except (TypeError, ValueError):
        top_episode_length = 0

    subseq_enabled_default = True
    subseq_enabled = bool(subseq_payload.get("enabled", subseq_enabled_default))
    if disable_subseq:
        subseq_enabled = False

    requested_episode_length = subseq_payload.get("episode_length", top_episode_length)
    try:
        requested_episode_length = int(requested_episode_length)
    except (TypeError, ValueError):
        requested_episode_length = 0

    if top_random_start is None:
        requested_random_start = bool(subseq_payload.get("random_start", subseq_enabled))
    else:
        requested_random_start = bool(top_random_start)

    max_episode_length = max(0, int(primary_sequence_length) - int(window_size) - 2)
    resolved_episode_length = 0
    if subseq_enabled and max_episode_length > 0:
        if requested_episode_length > 0:
            resolved_episode_length = min(int(requested_episode_length), max_episode_length)
        else:
            default_len = min(256, max_episode_length)
            if default_len >= max_episode_length and max_episode_length > 1:
                default_len = max(1, max_episode_length // 2)
            resolved_episode_length = int(default_len)

    resolved_random_start = bool(
        subseq_enabled and requested_random_start and resolved_episode_length > 0
    )

    resolved_subseq_enabled = bool(subseq_enabled and resolved_episode_length > 0)
    if resolved_subseq_enabled:
        logging.info(
            "Random-subsequence training enabled: random_start=%s episode_length=%d (max=%d).",
            resolved_random_start,
            int(resolved_episode_length),
            int(max_episode_length),
        )
    else:
        logging.info("Random-subsequence training disabled.")
    train_params["random_subsequence"] = {
        "enabled": resolved_subseq_enabled,
        "random_start": bool(resolved_random_start),
        "episode_length": int(resolved_episode_length),
        "max_episode_length": int(max_episode_length),
    }

    def _make_training_env():
        if len(train_datasets) == 1:
            dataset = train_datasets[0]
            return TradingEnv(
                df=dataset["df"],
                raw_df=dataset["raw_df"],
                window_size=window_size,
                commission_rate=train_params["commission_rate"],
                slippage=train_params["slippage"],
                symbol=dataset.get("symbol"),
                reward_config=reward_config,
                random_start=bool(resolved_random_start),
                episode_length=int(resolved_episode_length),
            )

        return MultiAssetTradingEnv(
            datasets=train_datasets,
            window_size=window_size,
            commission_rate=train_params["commission_rate"],
            slippage=train_params["slippage"],
            reward_config=reward_config,
            random_start=bool(resolved_random_start),
            episode_length=int(resolved_episode_length),
        )

    if n_envs > 1:
        env = make_vec_env(
            _make_training_env, n_envs=n_envs, seed=seed_value, vec_env_cls=DummyVecEnv
        )
        logging.info("Using vectorized training env: n_envs=%d (DummyVecEnv).", n_envs)
    else:
        env = _make_training_env()

    if early_stop_cfg.enabled and val_datasets:
        try:
            if len(val_datasets) == 1:
                val_dataset = val_datasets[0]
                val_env = TradingEnv(
                    df=val_dataset["df"],
                    raw_df=val_dataset["raw_df"],
                    window_size=window_size,
                    commission_rate=train_params["commission_rate"],
                    slippage=train_params["slippage"],
                    symbol=val_dataset.get("symbol"),
                    reward_config=reward_config,
                )
            else:
                val_env = MultiAssetTradingEnv(
                    datasets=val_datasets,
                    window_size=window_size,
                    commission_rate=train_params["commission_rate"],
                    slippage=train_params["slippage"],
                    reward_config=reward_config,
                )
            if hasattr(val_env, "random_start"):
                val_env.random_start = False
            if hasattr(val_env, "episode_length"):
                val_env.episode_length = 0
            logging.info("Validation environment ready for early-stop.")
        except Exception as exc:
            logging.warning("Early-stop disabled due to validation env failure: %s", exc)
            early_stop_cfg.enabled = False
            val_env = None

    if seed_value is not None and n_envs <= 1:
        try:
            env.reset(seed=seed_value)
        except Exception as exc:
            logging.warning("Failed to seed environment (seed=%s): %s", seed_value, exc)

    # 数据准备成功且环境创建完成后再创建数据库会话，避免空会话记录
    session_id = create_training_session(train_params["model_name"], train_params)
    if session_id is None:
        logging.error("无法创建训练会话，训练中止。")
        if progress_callback:
            progress_callback.status.emit("错误: 无法创建数据库会话")
        return

    if len(train_datasets) == 1:
        market_feature_dim = int(train_datasets[0]["df"].shape[1])
        sentiment_total_dim = int(
            sum(1 for col in train_datasets[0]["df"].columns if "sentiment_" in str(col))
        )
    else:
        market_feature_dim = int(sum(int(item["df"].shape[1]) for item in train_datasets))
        sentiment_total_dim = int(
            sum(
                sum(1 for col in item["df"].columns if "sentiment_" in str(col))
                for item in train_datasets
            )
        )
    aux_dim = max(0, int(env.observation_space.shape[1]) - int(market_feature_dim))

    # --- 模型训练 (总是创建新模型) ---
    model_name = train_params["model_name"]
    models_dir = ensure_dir(MODELS_DIR)
    model_path = models_dir / f"{model_name}.zip"
    tensorboard_dir = ensure_dir(TENSORBOARD_DIR)

    logging.info(f"Creating new model: {model_name}")
    # 选择策略网络
    if normalized_agent_type == "transformer":
        policy_class = TransformerActorCriticPolicy
        policy_kwargs = {
            "features_extractor_kwargs": {
                "features_dim": transformer_arch_config.get("features_dim", 128),
                "d_model": transformer_arch_config.get("d_model", 128),
                "n_heads": transformer_arch_config.get("n_heads", 4),
                "depth": transformer_arch_config.get("num_layers", 2),
                "ff_dim": transformer_arch_config.get("ff_dim"),
                "dropout": transformer_arch_config.get("dropout_rate", 0.1),
                "pooling": transformer_arch_config.get("pooling", "cls"),
                "positional_encoding": transformer_arch_config.get(
                    "positional_encoding", "learned"
                ),
                "use_relative_position_bias": transformer_arch_config.get(
                    "use_relative_position_bias", True
                ),
                "relative_attention_buckets": transformer_arch_config.get(
                    "relative_attention_buckets", 32
                ),
                "relative_attention_max_distance": transformer_arch_config.get(
                    "relative_attention_max_distance", 256
                ),
            }
        }
        logging.info("Using Transformer actor-critic policy for training.")
    elif normalized_agent_type == "cross_attention":
        policy_class = CrossAttentionActorCriticPolicy
        transformer_kwargs = cross_attention_arch_config.get("transformer_kwargs", {})
        policy_kwargs = {
            "features_extractor_kwargs": {
                "features_dim": cross_attention_arch_config.get("features_dim", 128),
                "cnn_dim": cross_attention_arch_config.get("cnn_dim", 128),
                "transformer_dim": cross_attention_arch_config.get("transformer_dim", 128),
                "cross_dim": cross_attention_arch_config.get("cross_dim", 128),
                "cross_heads": cross_attention_arch_config.get("cross_heads", 4),
                "transformer_kwargs": {
                    "d_model": transformer_kwargs.get(
                        "d_model", cross_attention_arch_config.get("transformer_dim", 128)
                    ),
                    "n_heads": transformer_kwargs.get(
                        "n_heads", cross_attention_arch_config.get("cross_heads", 4)
                    ),
                    "depth": transformer_kwargs.get("depth", 2),
                    "ff_dim": transformer_kwargs.get("ff_dim"),
                    "dropout": transformer_kwargs.get("dropout", 0.1),
                    "pooling": transformer_kwargs.get("pooling", "cls"),
                    "positional_encoding": transformer_kwargs.get("positional_encoding", "learned"),
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
                "sentiment_dim": sentiment_total_dim,
                "market_dim": market_feature_dim,
                "auxiliary_dim": aux_dim,
            }
        }
        logging.info("Using cross-attention hybrid policy for training.")
    else:
        policy_class = CNNActorCriticPolicy
        policy_kwargs = {
            "features_extractor_kwargs": {"features_dim": cnn_arch_config.get("features_dim", 128)}
        }
        logging.info("Using multiscale CNN actor-critic policy for training.")

    model = PPO(
        policy_class,
        env,
        verbose=0,
        learning_rate=agent_params["lr"],
        gamma=agent_params["gamma"],
        clip_range=agent_params["eps_clip"],
        n_epochs=agent_params["K_epochs"],
        n_steps=train_params["learning_frequency"],
        batch_size=agent_params.get("batch_size", 64),
        target_kl=agent_params.get("target_kl"),
        vf_coef=agent_params.get("vf_coef", 0.5),
        ent_coef=agent_params.get("ent_coef", 0.01),
        max_grad_norm=agent_params.get("max_grad_norm", 0.5),
        gae_lambda=agent_params.get("gae_lambda", 0.95),
        tensorboard_log=str(tensorboard_dir),
        device=device,
        seed=seed_value,
        policy_kwargs=policy_kwargs,
    )

    # --- 启动训练 ---
    # 使用从UI传入的回合数
    total_episodes = train_params["episodes"]
    subseq_cfg = (
        train_params.get("random_subsequence")
        if isinstance(train_params.get("random_subsequence"), dict)
        else {}
    )
    train_episode_length = int(subseq_cfg.get("episode_length") or 0)
    if bool(subseq_cfg.get("enabled")) and train_episode_length > 0:
        total_timesteps = train_episode_length * total_episodes
    else:
        total_timesteps = max(window_size, primary_sequence_length) * total_episodes

    callbacks = []
    if dashboard and progress_callback:
        callbacks.append(
            DashboardCallback(
                dashboard,
                progress_callback,
                total_timesteps,
                total_episodes,
                session_id,
                stop_flag_func=stop_flag_func,
            )
        )

    early_stop_impl = None
    best_checkpoint_path = None
    if early_stop_cfg.enabled and val_env is not None:
        inferred_eval_freq = int(train_params.get("learning_frequency") or 0)
        if inferred_eval_freq <= 0:
            inferred_eval_freq = int(primary_sequence_length)
        eval_freq_steps = (
            early_stop_cfg.eval_freq_steps
            if early_stop_cfg.eval_freq_steps > 0
            else inferred_eval_freq
        )
        if n_envs > 1:
            eval_freq_steps = int(eval_freq_steps) * int(n_envs)
        best_checkpoint_path = models_dir / f"{model_name}__best.zip"
        early_stop_impl = ValidationEarlyStopCallback(
            eval_env=val_env,
            config=early_stop_cfg,
            eval_freq_steps=eval_freq_steps,
            best_model_path=best_checkpoint_path,
        )
        callbacks.append(_SB3ValidationEarlyStopCallback(early_stop_impl))

    callback = None
    if callbacks:
        callback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)

    logging.info(f"Starting training for {total_timesteps} timesteps...")
    if progress_callback:
        progress_callback.status.emit("训练开始...")

    model.learn(total_timesteps=total_timesteps, callback=callback)

    logging.info("Training finished. Saving model...")
    if progress_callback:
        progress_callback.status.emit("训练完成。正在保存模型...")
        progress_callback.progress.emit(100)

    if early_stop_impl and early_stop_impl.best_model_path.exists():
        try:
            early_stop_impl.best_model_path.replace(model_path)
        except Exception:
            shutil.copy2(early_stop_impl.best_model_path, model_path)
            try:
                early_stop_impl.best_model_path.unlink(missing_ok=True)
            except Exception:
                logging.debug(
                    "Failed to remove temporary best model file: %s",
                    early_stop_impl.best_model_path,
                    exc_info=True,
                )
    else:
        model.save(str(model_path))

    # --- Send Email Notification ---
    def _truthy_env(name: str) -> bool:
        value = os.environ.get(name)
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _normalize_recipient_email(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate:
            return None
        # Backward-compat: dashboard used a placeholder email by default.
        if candidate.lower() == "example@mail.com":
            return None
        _, addr = parseaddr(candidate)
        if not addr or "@" not in addr:
            return None
        return addr

    user_config_path = USER_CONFIG_PATH
    if user_config_path.exists():
        with user_config_path.open("r", encoding="utf-8") as f:
            user_config = json.load(f)

        recipient_email = _normalize_recipient_email(user_config.get("email"))
        notify_enabled = bool(user_config.get("notify_on_training_finish", True))
        if _truthy_env("VIRGO_DISABLE_EMAIL") or _truthy_env("VIRGO_DISABLE_NOTIFICATIONS"):
            notify_enabled = False
        if bool(train_params.get("disable_email_notifications")) or bool(
            train_params.get("disable_notifications")
        ):
            notify_enabled = False

        if not notify_enabled:
            logging.info("Training-finish email notification disabled.")
            recipient_email = None
        if recipient_email:
            logging.info(f"正在向 {recipient_email} 发送训练完成通知...")
            subject = f"Virgo Trader 训练完成: {model_name}"

            # 格式化参数以便在邮件中清晰显示
            params_str = "\n".join([f"  - {key}: {value}" for key, value in train_params.items()])

            message = (
                f"您好,\n\n"
                f"模型 '{model_name}' 的训练已成功完成。\n\n"
                f"模型已保存至: {model_path.resolve()}\n\n"
                f"本次训练使用的参数如下:\n"
                f"{params_str}\n\n"
                f"祝您交易顺利！\n"
                f"Virgo Trader"
            )
            send_notification(recipient_email, subject, message)
        else:
            if notify_enabled:
                logging.warning("User email not found in config, skipping notification.")
    else:
        logging.warning(
            "user_config.json not found at %s, skipping notification.", user_config_path
        )


def main() -> None:
    """CLI entrypoint for running the training pipeline outside the desktop UI."""
    # --- Setup Logging ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Run Final Model Training")
    parser.add_argument(
        "--stock_code", type=str, default="510050.SH", help="Stock code to train on."
    )
    parser.add_argument(
        "--start_date", type=str, default="20210101", help="Start date for training data."
    )
    parser.add_argument(
        "--end_date", type=str, default="20231231", help="End date for training data."
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to train for.")
    parser.add_argument(
        "--commission_rate", type=float, default=0.0003, help="Commission rate for trades."
    )
    parser.add_argument("--slippage", type=float, default=0.0001, help="Slippage for trades.")
    parser.add_argument(
        "--agent_type",
        type=str,
        choices=["multiscale_cnn", "transformer", "cross_attention"],
        default="multiscale_cnn",
        help="Policy backbone to use.",
    )
    parser.add_argument(
        "--stock_pool", type=str, default=None, help="预设股票池键值，例如 'sse50'。"
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="Custom name for the saved model."
    )
    parser.add_argument(
        "--window_size", type=int, default=0, help="Sliding window size to override defaults."
    )
    parser.add_argument(
        "--learning_frequency",
        type=int,
        default=0,
        help="Number of env steps per policy update (n_steps).",
    )
    parser.add_argument(
        "--sentiment_dataset", type=str, default=None, help="Path to sentiment features JSONL."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="Enable validation-based early stopping (time-series split inside start/end).",
    )
    parser.add_argument(
        "--val_split_ratio", type=float, default=0.8, help="Train/val split ratio (default 0.8)."
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="calmar",
        choices=["calmar", "sharpe", "annual_return", "total_return", "final_value"],
        help="Validation metric to maximize for early stopping.",
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=5, help="Stop after N no-improve evals."
    )
    parser.add_argument(
        "--early_stop_min_evals", type=int, default=3, help="Minimum evals before stopping."
    )
    parser.add_argument(
        "--early_stop_eval_freq_steps",
        type=int,
        default=0,
        help="Evaluation frequency in env steps (0 => match PPO n_steps).",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum improvement to reset patience.",
    )

    args = parser.parse_args()

    # --- Construct train_params from args ---
    # Note: learning_frequency and window_size will be loaded from best_params.json
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.stock_pool:
        try:
            resolved_codes = list(get_stock_pool_codes(args.stock_pool))
            stock_pool_label = get_stock_pool_label(args.stock_pool)
        except Exception as exc:
            raise SystemExit(f"Failed to resolve stock pool '{args.stock_pool}': {exc}") from exc
    else:
        resolved_codes = [args.stock_code]
        stock_pool_label = None

    train_params = {
        "model_name": args.model_name or f"PPO_Final_Model_{timestamp}",
        "stock_pool": resolved_codes,
        "stock_pool_key": args.stock_pool,
        "stock_pool_label": stock_pool_label,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "commission_rate": args.commission_rate,
        "slippage": args.slippage,
        "episodes": args.episodes,
        "learning_frequency": max(0, args.learning_frequency),
        "window_size": max(0, args.window_size),
        "agent_type": args.agent_type,
        "sentiment_dataset": args.sentiment_dataset,
        "seed": args.seed,
    }
    if args.early_stop:
        train_params["early_stop"] = {
            "enabled": True,
            "val_split_ratio": float(args.val_split_ratio),
            "metric": args.early_stop_metric,
            "patience": int(args.early_stop_patience),
            "min_evals": int(args.early_stop_min_evals),
            "eval_freq_steps": int(args.early_stop_eval_freq_steps),
            "min_delta": float(args.early_stop_min_delta),
        }

    # --- Run Training ---
    # In a standalone script run, there's no dashboard or progress callback.
    train(train_params)


if __name__ == "__main__":
    main()
