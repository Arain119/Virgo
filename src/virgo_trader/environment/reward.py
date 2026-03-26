"""Reward computation for trading environments.

Defines configuration/weights and helper functions to combine multiple reward
signals (returns, drawdown, correlation penalties, etc.) into a single scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    """Weights for each reward component.

    Notes:
    - `correlation_penalty` is subtracted from the total reward.
    - `idle` multiplies `idle_penalty` (which is typically negative already).
    """

    dsr: float = 0.35
    simple_return: float = 0.20
    excess_return: float = 0.15
    drawdown: float = 0.15
    trend_consistency: float = 0.10
    correlation_penalty: float = 0.10
    idle: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, object] | None) -> "RewardWeights":
        base = cls()
        if not data:
            return base
        return cls(
            dsr=float(data.get("dsr", base.dsr)),
            simple_return=float(data.get("simple_return", base.simple_return)),
            excess_return=float(data.get("excess_return", base.excess_return)),
            drawdown=float(data.get("drawdown", base.drawdown)),
            trend_consistency=float(data.get("trend_consistency", base.trend_consistency)),
            correlation_penalty=float(data.get("correlation_penalty", base.correlation_penalty)),
            idle=float(data.get("idle", base.idle)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "dsr": float(self.dsr),
            "simple_return": float(self.simple_return),
            "excess_return": float(self.excess_return),
            "drawdown": float(self.drawdown),
            "trend_consistency": float(self.trend_consistency),
            "correlation_penalty": float(self.correlation_penalty),
            "idle": float(self.idle),
        }


@dataclass(frozen=True)
class RewardConfig:
    weights: RewardWeights = field(default_factory=RewardWeights)

    component_clip: float = 10.0
    reward_clip: float = 10.0

    drawdown_threshold: float = -0.05
    drawdown_scale: float = 2.0

    idle_penalty_scale: float = -0.0005
    idle_penalty_steps: int = 30

    corr_window_min_std: float = 1e-9
    dsr_min_std: float = 1e-8
    benchmark_log_return_floor: float = -0.999999
    positive_corr_only: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, object] | None) -> "RewardConfig":
        base = cls()
        if not data:
            return base

        weights = RewardWeights.from_dict(
            data.get("weights") if isinstance(data.get("weights"), dict) else None
        )
        return cls(
            weights=weights,
            component_clip=float(data.get("component_clip", base.component_clip)),
            reward_clip=float(data.get("reward_clip", base.reward_clip)),
            drawdown_threshold=float(data.get("drawdown_threshold", base.drawdown_threshold)),
            drawdown_scale=float(data.get("drawdown_scale", base.drawdown_scale)),
            idle_penalty_scale=float(data.get("idle_penalty_scale", base.idle_penalty_scale)),
            idle_penalty_steps=int(data.get("idle_penalty_steps", base.idle_penalty_steps)),
            corr_window_min_std=float(data.get("corr_window_min_std", base.corr_window_min_std)),
            dsr_min_std=float(data.get("dsr_min_std", base.dsr_min_std)),
            benchmark_log_return_floor=float(
                data.get("benchmark_log_return_floor", base.benchmark_log_return_floor)
            ),
            positive_corr_only=bool(data.get("positive_corr_only", base.positive_corr_only)),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "weights": self.weights.to_dict(),
            "component_clip": float(self.component_clip),
            "reward_clip": float(self.reward_clip),
            "drawdown_threshold": float(self.drawdown_threshold),
            "drawdown_scale": float(self.drawdown_scale),
            "idle_penalty_scale": float(self.idle_penalty_scale),
            "idle_penalty_steps": int(self.idle_penalty_steps),
            "corr_window_min_std": float(self.corr_window_min_std),
            "dsr_min_std": float(self.dsr_min_std),
            "benchmark_log_return_floor": float(self.benchmark_log_return_floor),
            "positive_corr_only": bool(self.positive_corr_only),
        }


def compute_composite_reward(
    *,
    log_return: float,
    simple_return: float,
    benchmark_return: float,
    portfolio_history: Sequence[float],
    portfolio_peak: float,
    dsr_a: float,
    dsr_b: float,
    dsr_eta: float,
    strategy_return_history: Sequence[float],
    asset_return_history: Sequence[float],
    corr_window: int,
    idle_steps: int,
    trend_consistency: float = 0.0,
    config: RewardConfig | None = None,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Compute a clipped composite reward and update DSR state.

    Returns:
        (reward, new_dsr_a, new_dsr_b, components)
    """

    if config is None:
        config = RewardConfig()

    log_return = float(np.nan_to_num(log_return, nan=0.0, posinf=0.0, neginf=0.0))
    simple_return = float(np.nan_to_num(simple_return, nan=0.0, posinf=0.0, neginf=0.0))
    benchmark_return = float(np.nan_to_num(benchmark_return, nan=0.0, posinf=0.0, neginf=0.0))
    trend_consistency = float(np.nan_to_num(trend_consistency, nan=0.0, posinf=0.0, neginf=0.0))

    # --- Differential Sharpe Ratio (DSR) component ---
    prev_a, prev_b = float(dsr_a), float(dsr_b)
    eta = float(dsr_eta) if dsr_eta is not None else 0.0
    if eta < 0.0:
        eta = 0.0
    if eta > 1.0:
        eta = 1.0

    new_a = (1.0 - eta) * prev_a + eta * log_return
    new_b = (1.0 - eta) * prev_b + eta * (log_return**2)

    d_a = new_a - prev_a
    d_b = new_b - prev_b
    std_dev = float(np.sqrt(max(new_b - new_a**2, config.dsr_min_std)))
    if std_dev != 0.0:
        dsr_reward = (d_a * std_dev - 0.5 * new_a * d_b / std_dev) / (std_dev**2)
    else:
        dsr_reward = 0.0
    dsr_reward = float(np.nan_to_num(dsr_reward, nan=0.0, posinf=0.0, neginf=0.0))
    dsr_reward = float(np.clip(dsr_reward, -config.component_clip, config.component_clip))

    # --- Excess return component ---
    if benchmark_return > config.benchmark_log_return_floor:
        benchmark_log_return = float(np.log(1.0 + benchmark_return))
    else:
        benchmark_log_return = 0.0
    excess_reward = float(log_return - benchmark_log_return)

    # --- Drawdown penalty component ---
    drawdown_reward = 0.0
    if portfolio_peak and portfolio_peak > 0 and portfolio_history:
        current_worth = float(portfolio_history[-1])
        current_drawdown = (current_worth - float(portfolio_peak)) / float(portfolio_peak)
        if current_drawdown < config.drawdown_threshold:
            drawdown_reward = float(current_drawdown * config.drawdown_scale)

    # --- Correlation penalty component ---
    corr_value = 0.0
    window = int(corr_window) if corr_window else 0
    if (
        window > 1
        and len(strategy_return_history) >= window
        and len(asset_return_history) >= window
    ):
        strat_slice = np.asarray(strategy_return_history[-window:], dtype=float)
        asset_slice = np.asarray(asset_return_history[-window:], dtype=float)
        strat_std = float(np.std(strat_slice))
        asset_std = float(np.std(asset_slice))
        if strat_std > config.corr_window_min_std and asset_std > config.corr_window_min_std:
            corr_matrix = np.corrcoef(strat_slice, asset_slice)
            if corr_matrix.size == 4:
                corr_value = float(corr_matrix[0, 1])
    corr_value = float(np.nan_to_num(corr_value, nan=0.0, posinf=0.0, neginf=0.0))
    correlation_penalty = max(0.0, corr_value) if config.positive_corr_only else abs(corr_value)

    # --- Idle penalty component ---
    idle_steps = int(idle_steps) if idle_steps is not None else 0
    if config.idle_penalty_steps > 0:
        idle_ratio = min(idle_steps / float(config.idle_penalty_steps), 1.0)
    else:
        idle_ratio = 0.0
    idle_penalty = float(config.idle_penalty_scale * idle_ratio)

    weights = config.weights
    reward = (
        weights.dsr * dsr_reward
        + weights.simple_return * simple_return
        + weights.excess_return * excess_reward
        + weights.drawdown * drawdown_reward
        + weights.trend_consistency * trend_consistency
        - weights.correlation_penalty * correlation_penalty
        + weights.idle * idle_penalty
    )

    reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
    reward = float(np.clip(reward, -config.reward_clip, config.reward_clip))

    components = {
        "dsr_reward": dsr_reward,
        "simple_return": simple_return,
        "excess_reward": excess_reward,
        "drawdown_reward": drawdown_reward,
        "trend_consistency": trend_consistency,
        "correlation": corr_value,
        "correlation_penalty": correlation_penalty,
        "idle_penalty": idle_penalty,
        "reward": reward,
    }
    return reward, float(new_a), float(new_b), components
