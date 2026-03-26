"""Dataset utilities to assemble features for training and backtesting.

This module focuses on loading sentiment datasets and merging them onto
feature-engineered market data without introducing lookahead bias.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .sentiment_loader import SentimentFeatureLoader


def load_sentiment_loader(
    dataset_path: Optional[str | Path],
) -> Tuple[Optional[SentimentFeatureLoader], List[str]]:
    """
    Load sentiment dataset if provided.

    Returns:
        (loader, columns). If loading fails, returns (None, []) and logs a warning.
    """
    if not dataset_path:
        return None, []

    try:
        loader = SentimentFeatureLoader(dataset_path)
    except Exception as exc:
        logging.warning("Failed to load sentiment dataset %s: %s", dataset_path, exc)
        return None, []

    return loader, loader.feature_columns


def attach_sentiment_features(
    processed_df: pd.DataFrame,
    symbol: str,
    sentiment_loader: Optional[SentimentFeatureLoader],
    sentiment_columns: List[str],
) -> pd.DataFrame:
    """
    Attach sentiment features to an existing feature-engineered dataframe.

    Key behaviors:
    - Reindex sentiment data onto trading days using as-of forward fill. This
      preserves weekend/non-trading-day news by carrying it into the next
      available trading day.
    - Ensures a stable set/order of sentiment columns (missing columns -> 0).
    """
    if processed_df is None or processed_df.empty:
        return processed_df
    if not sentiment_columns:
        return processed_df
    if sentiment_loader is None:
        for column in sentiment_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0.0
        non_sentiment_cols = [col for col in processed_df.columns if col not in sentiment_columns]
        return processed_df[non_sentiment_cols + sentiment_columns]

    trade_index = processed_df.index
    if not isinstance(trade_index, pd.DatetimeIndex):
        trade_index = pd.to_datetime(trade_index)

    sentiment_df = sentiment_loader.get_symbol_frame(symbol)
    if (
        sentiment_df is not None
        and not sentiment_df.empty
        and isinstance(sentiment_df.index, pd.DatetimeIndex)
    ):
        # Avoid leaking old sentiment into future windows: only keep sentiment data that can be mapped
        # to the current trading index (with a small calendar buffer for weekend/holiday carry-over).
        min_ts = trade_index.min() - pd.Timedelta(days=7)
        max_ts = trade_index.max()
        sentiment_df = sentiment_df.loc[
            (sentiment_df.index >= min_ts) & (sentiment_df.index <= max_ts)
        ]
    if sentiment_df is None or sentiment_df.empty:
        for column in sentiment_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0.0
        non_sentiment_cols = [col for col in processed_df.columns if col not in sentiment_columns]
        return processed_df[non_sentiment_cols + sentiment_columns]

    base_cols = ["sentiment_score", "sentiment_magnitude", "sentiment_volume"]
    missing_bases = [col for col in base_cols if col not in sentiment_df.columns]
    if missing_bases:
        logging.warning(
            "Sentiment frame for %s missing base cols %s; filling zeros.",
            symbol,
            ",".join(missing_bases),
        )
        for col in missing_bases:
            sentiment_df[col] = 0.0

    base_daily = sentiment_df[base_cols].copy()
    trade_values = trade_index.values
    mapped_days = []
    for date_value in base_daily.index:
        pos = trade_index.searchsorted(date_value)
        if pos >= len(trade_values):
            continue
        mapped_days.append(trade_values[pos])

    if not mapped_days:
        for column in sentiment_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0.0
        non_sentiment_cols = [col for col in processed_df.columns if col not in sentiment_columns]
        return processed_df[non_sentiment_cols + sentiment_columns]

    mapped = base_daily.iloc[: len(mapped_days)].copy()
    mapped["trade_date"] = pd.to_datetime(mapped_days)

    mapped["sentiment_volume"] = mapped["sentiment_volume"].astype(float)
    mapped["weighted_score"] = mapped["sentiment_score"].astype(float) * mapped["sentiment_volume"]
    mapped["weighted_magnitude"] = (
        mapped["sentiment_magnitude"].astype(float) * mapped["sentiment_volume"]
    )

    grouped = mapped.groupby("trade_date", sort=True)
    volume = grouped["sentiment_volume"].sum()
    weighted_den = volume.where(volume > 0, 1.0)
    score = grouped["weighted_score"].sum() / weighted_den
    magnitude = grouped["weighted_magnitude"].sum() / weighted_den

    aligned_base = (
        pd.DataFrame(
            {
                "sentiment_score": score,
                "sentiment_magnitude": magnitude,
                "sentiment_volume": volume,
            }
        )
        .reindex(trade_index)
        .fillna(0.0)
    )

    # Recompute multiscale features on trading days so that "no news" days are represented as 0 volume.
    aligned_full = aligned_base.copy()
    ema_windows = getattr(sentiment_loader, "ema_windows", (2, 5, 10))
    for base in base_cols:
        series = aligned_base[base]
        for window in ema_windows:
            ema_col = f"{base}_ema_{window}"
            roc_col = f"{base}_roc_{window}"
            aligned_full[ema_col] = series.ewm(span=window, adjust=False, min_periods=1).mean()
            pct = series.pct_change(window).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            aligned_full[roc_col] = pct.clip(-5.0, 5.0)

    aligned_full["sentiment_score_energy"] = (
        aligned_base["sentiment_score"].pow(2).rolling(5, min_periods=1).mean().fillna(0.0)
    )

    processed_df = processed_df.join(aligned_full, how="left")

    for column in sentiment_columns:
        if column not in processed_df.columns:
            processed_df[column] = 0.0

    processed_df[sentiment_columns] = processed_df[sentiment_columns].fillna(0.0)
    non_sentiment_cols = [col for col in processed_df.columns if col not in sentiment_columns]
    return processed_df[non_sentiment_cols + sentiment_columns]
