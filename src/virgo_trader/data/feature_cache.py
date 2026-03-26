"""Disk cache for feature-engineered datasets.

The cache stores pre-computed pandas DataFrames to speed up repeated training
and backtest runs for the same (symbol, date range, feature flags).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ..utils.paths import FEATURE_CACHE_DIR, ensure_dir

logger = logging.getLogger(__name__)


def _safe_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper().replace(".", "_").replace("/", "_")


def _cal_tag(use_calendar_features: bool) -> str:
    return "cal1" if use_calendar_features else "cal0"


def feature_cache_path(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    use_calendar_features: bool,
) -> Path:
    safe_symbol = _safe_symbol(symbol)
    cal_tag = _cal_tag(use_calendar_features)
    filename = f"{safe_symbol}__{start_date}_{end_date}__{cal_tag}.pkl"
    return ensure_dir(FEATURE_CACHE_DIR) / filename


def load_cached_features(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        # Local trusted cache only; never load untrusted pickle files.
        df = pd.read_pickle(path)  # noqa: S301
    except Exception as exc:
        logger.warning("Failed to read feature cache %s: %s", path, exc)
        return None
    if df is None or getattr(df, "empty", True):
        return None
    return df


def save_cached_features(path: Path, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path)
    except Exception as exc:
        logger.warning("Failed to write feature cache %s: %s", path, exc)


def _parse_feature_cache_filename(path: Path) -> Optional[Tuple[str, str, str]]:
    """
    Returns (start_date, end_date, cal_tag) parsed from file stem.

    Expected filename pattern:
        SYMBOL__YYYYMMDD_YYYYMMDD__cal0.pkl
    """
    stem = path.stem
    parts = stem.split("__")
    if len(parts) != 3:
        return None
    _, date_part, cal_tag = parts
    if "_" not in date_part:
        return None
    try:
        start_date, end_date = date_part.split("_", 1)
    except ValueError:
        return None
    if not start_date or not end_date or not cal_tag:
        return None
    return start_date, end_date, cal_tag


def find_superset_feature_cache(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    use_calendar_features: bool,
) -> Optional[Path]:
    """
    Find a cached feature file that fully covers the requested date range.

    This enables reusing a "long range" precomputed feature cache for shorter
    training spans, avoiding recomputing pandas-ta indicators repeatedly.
    """
    base_dir = ensure_dir(FEATURE_CACHE_DIR)
    safe_symbol = _safe_symbol(symbol)
    cal_tag = _cal_tag(use_calendar_features)

    requested_start = pd.to_datetime(start_date, errors="coerce")
    requested_end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(requested_start) or pd.isna(requested_end):
        return None

    candidates = list(base_dir.glob(f"{safe_symbol}__*__{cal_tag}.pkl"))
    best_path: Optional[Path] = None
    best_span_days: Optional[int] = None

    for path in candidates:
        parsed = _parse_feature_cache_filename(path)
        if parsed is None:
            continue
        cached_start_str, cached_end_str, cached_tag = parsed
        if cached_tag != cal_tag:
            continue

        cached_start = pd.to_datetime(cached_start_str, errors="coerce")
        cached_end = pd.to_datetime(cached_end_str, errors="coerce")
        if pd.isna(cached_start) or pd.isna(cached_end):
            continue

        if cached_start <= requested_start and cached_end >= requested_end:
            span_days = int((cached_end - cached_start).days)
            if best_span_days is None or span_days < best_span_days:
                best_span_days = span_days
                best_path = path

    return best_path


def load_cached_features_for_range(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    use_calendar_features: bool,
    write_back: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Load a feature cache for the requested range, falling back to a superset cache.

    If a superset cache is found, we slice it to the requested range and (optionally)
    persist the slice as the exact cache file for faster subsequent loads.
    """
    target_path = feature_cache_path(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        use_calendar_features=use_calendar_features,
    )
    direct = load_cached_features(target_path)
    if direct is not None:
        return direct

    superset_path = find_superset_feature_cache(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        use_calendar_features=use_calendar_features,
    )
    if superset_path is None:
        return None

    superset_df = load_cached_features(superset_path)
    if superset_df is None:
        return None

    start_ts = pd.to_datetime(start_date, errors="coerce")
    end_ts = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return None

    df = superset_df
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            return None
    df = df.sort_index()

    sliced = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    if sliced is None or getattr(sliced, "empty", True):
        return None

    logger.info(
        "Feature cache fallback hit for %s: %s -> %s", symbol, superset_path.name, target_path.name
    )
    if write_back:
        save_cached_features(target_path, sliced)
    return sliced
