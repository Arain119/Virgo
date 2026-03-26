"""Unit tests for early stopping utilities."""

import pandas as pd

from virgo_trader.utils.early_stopping import (
    compute_portfolio_metrics,
    score_from_metrics,
    split_datasets_by_ratio,
)


def test_compute_portfolio_metrics_basic():
    metrics = compute_portfolio_metrics([100.0, 110.0, 105.0, 120.0])
    assert metrics["final_value"] == 120.0
    assert abs(metrics["total_return"] - 0.2) < 1e-12
    assert metrics["max_drawdown"] < 0
    assert "sharpe" in metrics
    assert "calmar" in metrics


def test_score_from_metrics_respects_choice():
    metrics = {"calmar": 1.5, "sharpe": 2.0, "final_value": 123.0}
    assert score_from_metrics(metrics, "sharpe") == 2.0
    assert score_from_metrics(metrics, "final_value") == 123.0
    assert score_from_metrics(metrics, "unknown") == 1.5


def test_split_datasets_by_ratio_uses_common_index():
    index = pd.date_range("2023-01-01", periods=100, freq="B")
    df_a = pd.DataFrame({"close": range(100)}, index=index)
    raw_a = pd.DataFrame({"close": range(100)}, index=index)
    df_b = pd.DataFrame({"close": range(100)}, index=index)
    raw_b = pd.DataFrame({"close": range(100)}, index=index)

    datasets = [
        {"symbol": "AAA", "df": df_a, "raw_df": raw_a},
        {"symbol": "BBB", "df": df_b, "raw_df": raw_b},
    ]

    train_sets, val_sets, info = split_datasets_by_ratio(datasets, window_size=12, split_ratio=0.8)
    assert len(train_sets) == len(datasets)
    assert len(val_sets) == len(datasets)
    assert info["train_rows"] + info["val_rows"] == len(index)
    assert info["train_rows"] > 12 + 2
    assert info["val_rows"] > 12 + 2
