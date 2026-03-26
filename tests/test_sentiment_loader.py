"""Unit tests for sentiment loader utilities."""

from __future__ import annotations

import json

import pandas as pd

from virgo_trader.data.dataset_builder import attach_sentiment_features, load_sentiment_loader
from virgo_trader.data.sentiment_loader import SentimentFeatureLoader


def _write_jsonl(path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_get_symbol_frame_supports_dot_and_prefix_formats(tmp_path) -> None:
    dataset = tmp_path / "sentiment.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "symbol": "SH600519",
                "published_at": "2024-01-02T01:00:00Z",
                "intensities": [7, 6, 2],
                "labels": ["pos"],
            },
            {
                "symbol": "SZ000001",
                "published_at": "2024-01-02T02:00:00Z",
                "intensities": [1, 2, 3],
                "labels": ["neg"],
            },
        ],
    )

    loader = SentimentFeatureLoader(dataset)
    assert len(loader.feature_columns) == 22

    sh_frame = loader.get_symbol_frame("600519.SH")
    assert sh_frame is not None
    assert not sh_frame.empty
    assert set(loader.feature_columns).issubset(set(sh_frame.columns))

    sh_frame2 = loader.get_symbol_frame("SH600519")
    assert sh_frame2 is not None
    pd.testing.assert_index_equal(sh_frame.index, sh_frame2.index)

    # Best-effort numeric code matching should return one of the exchanges.
    numeric_frame = loader.get_symbol_frame("600519")
    assert numeric_frame is not None
    assert not numeric_frame.empty


def test_attach_sentiment_features_carries_weekend_news_forward(tmp_path) -> None:
    dataset = tmp_path / "sentiment.jsonl"
    # Weekend news (Saturday) should be carried into the next trading day when aligning to trading dates.
    _write_jsonl(
        dataset,
        [
            {
                "symbol": "SH600519",
                "published_at": "2023-01-07T01:00:00Z",
                "intensities": [7],
                "labels": ["pos"],
            },
        ],
    )

    sentiment_loader, sentiment_columns = load_sentiment_loader(dataset)
    assert sentiment_loader is not None
    assert len(sentiment_columns) == 22

    trading_days = pd.to_datetime(["2023-01-06", "2023-01-09"])  # Fri, Mon
    processed_df = pd.DataFrame({"dummy": [1.0, 2.0]}, index=trading_days)

    enriched = attach_sentiment_features(
        processed_df=processed_df,
        symbol="600519.SH",
        sentiment_loader=sentiment_loader,
        sentiment_columns=sentiment_columns,
    )

    assert "sentiment_score" in enriched.columns
    assert enriched.loc[pd.Timestamp("2023-01-06"), "sentiment_score"] == 0.0
    assert enriched.loc[pd.Timestamp("2023-01-09"), "sentiment_score"] > 0.0


def test_attach_sentiment_features_shifts_intraday_news_to_next_trading_day(tmp_path) -> None:
    dataset = tmp_path / "sentiment.jsonl"
    # News during trading hours (>= 09:30 Shanghai) should be shifted to the next trading day to avoid leakage.
    _write_jsonl(
        dataset,
        [
            {
                "symbol": "SH600519",
                "published_at": "2023-01-03T02:00:00Z",  # 10:00 Asia/Shanghai
                "intensities": [7],
                "labels": ["pos"],
            },
        ],
    )

    sentiment_loader, sentiment_columns = load_sentiment_loader(dataset)
    assert sentiment_loader is not None
    assert len(sentiment_columns) == 22

    trading_days = pd.to_datetime(["2023-01-03", "2023-01-04"])  # Tue, Wed
    processed_df = pd.DataFrame({"dummy": [1.0, 2.0]}, index=trading_days)

    enriched = attach_sentiment_features(
        processed_df=processed_df,
        symbol="600519.SH",
        sentiment_loader=sentiment_loader,
        sentiment_columns=sentiment_columns,
    )

    assert enriched.loc[pd.Timestamp("2023-01-03"), "sentiment_score"] == 0.0
    assert enriched.loc[pd.Timestamp("2023-01-04"), "sentiment_score"] > 0.0
