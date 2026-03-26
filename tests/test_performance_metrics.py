"""Unit tests for performance metric calculations."""

from __future__ import annotations

import pandas as pd

from virgo_trader.utils.performance_metrics import calculate_performance_metrics


def test_calculate_performance_metrics_handles_multi_asset_trades() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    portfolio = pd.Series([100.0, 101.0, 100.0, 102.0], index=dates)

    trades = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01",
                "side": "BUY",
                "price": 10.0,
                "quantity": 10.0,
                "symbol": "AAA",
            },
            {
                "timestamp": "2024-01-03",
                "side": "SELL",
                "price": 11.0,
                "quantity": 10.0,
                "symbol": "AAA",
            },  # +10
            {
                "timestamp": "2024-01-02",
                "side": "BUY",
                "price": 20.0,
                "quantity": 5.0,
                "symbol": "BBB",
            },
            {
                "timestamp": "2024-01-04",
                "side": "SELL",
                "price": 18.0,
                "quantity": 5.0,
                "symbol": "BBB",
            },  # -10
        ]
    )
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades = trades.set_index("timestamp").sort_index()

    kline = pd.DataFrame(
        {
            "close": [10.0, 10.5, 10.3, 10.8],
            "high": [10.2, 10.6, 10.4, 10.9],
            "low": [9.8, 10.1, 10.0, 10.5],
        },
        index=dates,
    )

    metrics = calculate_performance_metrics(
        portfolio_history=portfolio, trades=trades, kline_data=kline
    )

    assert metrics["总交易次数"] == 2
    assert metrics["胜率"] == "50.00%"
    assert metrics["盈亏比"] == "1.00"
    assert metrics["平均持仓时间"] == "2.0天"
