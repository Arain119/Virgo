"""Unit tests for reward calculation utilities."""

from __future__ import annotations

import numpy as np

from virgo_trader.environment.reward import RewardConfig, compute_composite_reward


def test_compute_composite_reward_updates_dsr_state_and_returns_finite() -> None:
    config = RewardConfig()
    reward, new_a, new_b, components = compute_composite_reward(
        log_return=0.01,
        simple_return=0.01,
        benchmark_return=0.0,
        portfolio_history=[100.0, 101.0],
        portfolio_peak=101.0,
        dsr_a=0.0,
        dsr_b=0.0,
        dsr_eta=1.0 / 252.0,
        strategy_return_history=[0.0, 0.01],
        asset_return_history=[0.0, 0.0],
        corr_window=2,
        idle_steps=0,
        trend_consistency=0.0,
        config=config,
    )

    assert np.isfinite(reward)
    assert np.isfinite(new_a)
    assert np.isfinite(new_b)
    assert "reward" in components


def test_compute_composite_reward_positive_corr_penalty_only() -> None:
    config = RewardConfig(positive_corr_only=True)
    reward, _, _, components = compute_composite_reward(
        log_return=0.0,
        simple_return=0.0,
        benchmark_return=0.0,
        portfolio_history=[100.0, 100.0],
        portfolio_peak=100.0,
        dsr_a=0.0,
        dsr_b=0.0,
        dsr_eta=1.0 / 252.0,
        strategy_return_history=[0.01, -0.01],
        asset_return_history=[-0.01, 0.01],  # corr ~= -1
        corr_window=2,
        idle_steps=0,
        trend_consistency=0.0,
        config=config,
    )

    assert components["correlation"] < 0.0
    assert components["correlation_penalty"] == 0.0
    assert np.isfinite(reward)
