"""Early stopping utilities for PPO training.

Provides validation splitting, metric computation, and callback helpers to stop
training when performance plateaus.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopConfig:
    enabled: bool = False
    val_split_ratio: float = 0.8
    eval_freq_steps: int = 0  # 0 -> infer from train sequence length
    min_evals: int = 3
    patience: int = 5
    min_delta: float = 0.0
    metric: str = "calmar"  # calmar|sharpe|annual_return|total_return|final_value
    deterministic: bool = True

    @classmethod
    def from_payload(cls, payload: object) -> "EarlyStopConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return cls()
        if not isinstance(payload, Mapping):
            return cls()

        base = cls()
        metric = str(payload.get("metric", base.metric)).strip().lower()
        if metric not in {"calmar", "sharpe", "annual_return", "total_return", "final_value"}:
            metric = base.metric

        try:
            val_split_ratio = float(payload.get("val_split_ratio", base.val_split_ratio))
        except (TypeError, ValueError):
            val_split_ratio = base.val_split_ratio
        if not (0.0 < val_split_ratio < 1.0):
            val_split_ratio = base.val_split_ratio

        def _to_int(key: str, default: int) -> int:
            try:
                value = int(payload.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(0, value)

        def _to_float(key: str, default: float) -> float:
            try:
                value = float(payload.get(key, default))
            except (TypeError, ValueError):
                value = default
            return float(value)

        return cls(
            enabled=bool(payload.get("enabled", base.enabled)),
            val_split_ratio=val_split_ratio,
            eval_freq_steps=_to_int("eval_freq_steps", base.eval_freq_steps),
            min_evals=max(0, _to_int("min_evals", base.min_evals)),
            patience=max(1, _to_int("patience", base.patience)),
            min_delta=_to_float("min_delta", base.min_delta),
            metric=metric,
            deterministic=bool(payload.get("deterministic", base.deterministic)),
        )


def compute_portfolio_metrics(
    values: Sequence[float], *, trading_days_per_year: int = 252
) -> Dict[str, float]:
    series = np.asarray(list(values), dtype=np.float64)
    if series.size < 2:
        return {
            "final_value": float(series[-1]) if series.size else 0.0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
        }

    initial = float(series[0])
    final = float(series[-1])
    total_return = (final / initial - 1.0) if initial > 0 else 0.0

    periods = int(series.size - 1)
    annual_return = 0.0
    if periods > 0 and (1.0 + total_return) > 0:
        annual_return = float(
            (1.0 + total_return) ** (float(trading_days_per_year) / float(periods)) - 1.0
        )

    daily_returns = np.diff(series) / np.where(series[:-1] == 0, 1.0, series[:-1])
    daily_std = float(np.std(daily_returns)) if daily_returns.size else 0.0
    if daily_returns.size and daily_std > 1e-12:
        sharpe = float(np.mean(daily_returns) / daily_std * np.sqrt(trading_days_per_year))
    else:
        sharpe = 0.0

    cummax = np.maximum.accumulate(series)
    drawdown = (series - cummax) / np.where(cummax == 0, 1.0, cummax)
    max_drawdown = float(np.min(drawdown)) if drawdown.size else 0.0

    dd = abs(max_drawdown)
    calmar = float(annual_return / max(dd, 1e-12)) if dd >= 0 else 0.0

    return {
        "final_value": final,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
    }


def score_from_metrics(metrics: Mapping[str, float], metric: str) -> float:
    key = (metric or "").strip().lower()
    if key in {"calmar", "sharpe", "annual_return", "total_return", "final_value"}:
        return float(metrics.get(key, 0.0))
    return float(metrics.get("calmar", 0.0))


def split_datasets_by_ratio(
    datasets: Sequence[Dict[str, object]],
    *,
    window_size: int,
    split_ratio: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """
    Split multi-asset datasets by a time-ordered ratio using a common intersection
    index, preventing lookahead leakage between train and validation segments.
    """
    if not datasets:
        raise ValueError("datasets is empty.")

    common_index = None
    for item in datasets:
        df = item.get("df")
        raw_df = item.get("raw_df")
        if df is None or raw_df is None:
            raise ValueError("Each dataset item must have 'df' and 'raw_df'.")
        df_index = getattr(df, "index", None)
        raw_index = getattr(raw_df, "index", None)
        if df_index is None or raw_index is None:
            raise ValueError("Dataset 'df'/'raw_df' must have an index.")
        item_index = df_index.intersection(raw_index)
        common_index = item_index if common_index is None else common_index.intersection(item_index)

    if common_index is None or len(common_index) <= (window_size + 2):
        raise ValueError("Common index too short for a train/val split.")

    common_index = common_index.sort_values()
    split_point = int(len(common_index) * float(split_ratio))
    split_point = max(window_size + 2, min(len(common_index) - (window_size + 2), split_point))

    train_index = common_index[:split_point]
    val_index = common_index[split_point:]
    if len(train_index) <= (window_size + 2) or len(val_index) <= (window_size + 2):
        raise ValueError("Split produced insufficient data for train or validation.")

    train_datasets: List[Dict[str, object]] = []
    val_datasets: List[Dict[str, object]] = []
    for item in datasets:
        df = item["df"]
        raw_df = item["raw_df"]
        symbol = item.get("symbol")
        train_datasets.append(
            {"symbol": symbol, "df": df.loc[train_index], "raw_df": raw_df.loc[train_index]}
        )
        val_datasets.append(
            {"symbol": symbol, "df": df.loc[val_index], "raw_df": raw_df.loc[val_index]}
        )

    split_info = {
        "split_ratio": float(split_ratio),
        "train_start": str(train_index[0]),
        "train_end": str(train_index[-1]),
        "val_start": str(val_index[0]),
        "val_end": str(val_index[-1]),
        "train_rows": int(len(train_index)),
        "val_rows": int(len(val_index)),
    }
    return train_datasets, val_datasets, split_info


class ValidationEarlyStopCallback:
    """
    Stable-Baselines3 callback-like object (duck-typed) for validation-based early stopping.

    We intentionally avoid importing stable_baselines3 at module import time to keep import
    side effects minimal. The training pipeline wires it up by subclassing BaseCallback.
    """

    def __init__(
        self,
        *,
        eval_env,
        config: EarlyStopConfig,
        eval_freq_steps: int,
        best_model_path: Path,
        verbose: int = 0,
    ):
        self.eval_env = eval_env
        self.config = config
        self.eval_freq_steps = max(1, int(eval_freq_steps))
        self.best_model_path = Path(best_model_path)
        self.verbose = int(verbose)

        self.evals_done = 0
        self.best_score = -float("inf")
        self.no_improve_evals = 0
        self.last_metrics: Dict[str, float] = {}

        self._next_eval_timestep = self.eval_freq_steps

    def _evaluate(self, model) -> Tuple[float, Dict[str, float]]:
        obs, _info = self.eval_env.reset()
        done = False
        while not done:
            action, _state = model.predict(obs, deterministic=bool(self.config.deterministic))
            obs, _reward, terminated, truncated, _info = self.eval_env.step(action)
            done = bool(terminated) or bool(truncated)

        values = getattr(self.eval_env, "portfolio_history", None)
        metrics = compute_portfolio_metrics(values or [])
        score = score_from_metrics(metrics, self.config.metric)
        return score, metrics

    def should_eval(self, num_timesteps: int) -> bool:
        return int(num_timesteps) >= self._next_eval_timestep

    def on_eval(self, model, num_timesteps: int) -> bool:
        score, metrics = self._evaluate(model)
        self.last_metrics = metrics
        self.evals_done += 1
        improved = score > (self.best_score + float(self.config.min_delta))

        if improved:
            self.best_score = float(score)
            self.no_improve_evals = 0
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(self.best_model_path))
            logger.info(
                "[early-stop] new best %.6f (%s=%s, sharpe=%.3f, max_dd=%.3f, ann=%.3f) @ step=%d",
                self.best_score,
                self.config.metric,
                round(score, 6),
                metrics.get("sharpe", 0.0),
                metrics.get("max_drawdown", 0.0),
                metrics.get("annual_return", 0.0),
                int(num_timesteps),
            )
        else:
            if self.evals_done >= int(self.config.min_evals):
                self.no_improve_evals += 1
            logger.info(
                "[early-stop] eval %.6f (best=%.6f, no_improve=%d/%d, sharpe=%.3f, max_dd=%.3f) @ step=%d",
                float(score),
                float(self.best_score),
                int(self.no_improve_evals),
                int(self.config.patience),
                metrics.get("sharpe", 0.0),
                metrics.get("max_drawdown", 0.0),
                int(num_timesteps),
            )

        self._next_eval_timestep += self.eval_freq_steps

        if self.evals_done >= int(self.config.min_evals) and self.no_improve_evals >= int(
            self.config.patience
        ):
            logger.info(
                "[early-stop] stopping: no improvement for %d evals.", int(self.no_improve_evals)
            )
            return False
        return True
