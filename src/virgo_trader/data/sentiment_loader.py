"""Load and aggregate news sentiment features for trading models.

The loader reads a JSONL dataset and produces per-symbol, per-day (or per-bar)
sentiment features that can be aligned onto trading calendars.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import time as dt_time
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.paths import DATA_ROOT


def _normalize_score(intensities: List[float]) -> float:
    if not intensities:
        return 0.0
    arr = np.asarray(intensities, dtype=np.float32)
    normalized = (arr - 4.0) / 3.0  # map [1,7] -> [-1,1]
    return float(np.clip(normalized.mean(), -1.0, 1.0))


def _normalize_magnitude(intensities: List[float]) -> float:
    if not intensities:
        return 0.0
    arr = np.asarray(intensities, dtype=np.float32)
    normalized = np.abs((arr - 4.0) / 3.0)
    return float(np.clip(normalized.mean(), 0.0, 1.0))


@dataclass
class SentimentFeatureSet:
    frames: Dict[str, pd.DataFrame]
    columns: List[str]
    embedding_dim: int


class SentimentFeatureLoader:
    """
    将 JSONL 形式的情绪推理结果聚合为日级别的情绪特征，并自动生成多尺度特征供模型自适应学习。

    为避免信息泄露：默认以交易日开盘时间（09:30）作为“可交易信息”的截断点，
    当新闻发布时间 >= 截断点时，会被归入下一自然日（随后在与交易日对齐时映射到下一交易日）。
    """

    def __init__(
        self,
        dataset_path: str | Path,
        ema_windows: Optional[Sequence[int]] = None,
        news_cutoff_time: dt_time | str | None = None,
    ) -> None:
        self.path = Path(dataset_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Sentiment dataset not found: {dataset_path}")
        windows = ema_windows or (2, 5, 10)
        valid_windows = sorted(
            {int(w) for w in windows if isinstance(w, (int, float)) and int(w) > 1}
        )
        self.ema_windows = tuple(valid_windows) if valid_windows else (2, 5, 10)
        self.news_cutoff_time = _parse_cutoff_time(news_cutoff_time)
        self._feature_set = self._build_feature_set()

    @property
    def feature_columns(self) -> List[str]:
        return self._feature_set.columns

    @property
    def embedding_dim(self) -> int:
        return self._feature_set.embedding_dim

    def get_symbol_frame(self, symbol: str) -> Optional[pd.DataFrame]:
        if not symbol:
            return None

        frames = self._feature_set.frames
        raw = str(symbol).strip()
        if not raw:
            return None

        # 1) Direct match (fast path)
        frame = frames.get(raw)
        if frame is not None:
            return frame

        upper = raw.upper()
        frame = frames.get(upper)
        if frame is not None:
            return frame

        # 2) Common A-share symbol format conversions
        #    - Training/simulation may use '600519.SH' while sentiment exports use 'SH600519'
        if "." in upper:
            core, market = upper.split(".", 1)
            alt = f"{market}{core}"
            frame = frames.get(alt)
            if frame is not None:
                return frame
        elif upper.startswith(("SH", "SZ")) and len(upper) > 2:
            market = upper[:2]
            core = upper[2:]
            alt = f"{core}.{market}"
            frame = frames.get(alt)
            if frame is not None:
                return frame

        # 3) Bare numeric codes: try both exchanges (best-effort)
        if upper.isdigit() and len(upper) in {5, 6}:
            candidates = (
                f"SH{upper}",
                f"SZ{upper}",
                f"{upper}.SH",
                f"{upper}.SZ",
            )
            for cand in candidates:
                frame = frames.get(cand)
                if frame is not None:
                    return frame

        return None

    def available_symbols(self) -> Iterable[str]:
        return self._feature_set.frames.keys()

    def _build_feature_set(self) -> SentimentFeatureSet:
        records: List[Dict[str, object]] = []
        embedding_dim = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                symbol = payload.get("symbol")
                published_at = payload.get("published_at")
                if not symbol or not published_at:
                    continue

                timestamp = pd.to_datetime(published_at, utc=True, errors="coerce")
                if pd.isna(timestamp):
                    continue
                local_ts = timestamp.tz_convert("Asia/Shanghai")
                effective_date = local_ts.date()
                if local_ts.time() >= self.news_cutoff_time:
                    effective_date = effective_date + timedelta(days=1)

                intensities = payload.get("intensities") or []
                if isinstance(intensities, list):
                    intensities = [float(x) for x in intensities if x is not None]
                else:
                    intensities = []

                record = {
                    "symbol": symbol,
                    "date": pd.to_datetime(effective_date),
                    "score": _normalize_score(intensities),
                    "magnitude": _normalize_magnitude(intensities),
                    "volume": float(len(intensities)) if intensities else 1.0,
                }

                embedding = payload.get("embedding") or payload.get("vector")
                if isinstance(embedding, list) and embedding:
                    emb_array = np.asarray(embedding, dtype=np.float32)
                    embedding_dim = max(embedding_dim, emb_array.size)
                    record["embedding"] = emb_array
                records.append(record)

        if not records:
            return SentimentFeatureSet(frames={}, columns=[], embedding_dim=0)

        df = pd.DataFrame(records)

        aggregation_rows: List[Dict[str, object]] = []
        for (symbol, date), group in df.groupby(["symbol", "date"]):
            weight = group["volume"].sum()
            weight = float(weight) if weight and weight > 0 else 1.0

            score = float((group["score"] * group["volume"]).sum() / weight)
            magnitude = float((group["magnitude"] * group["volume"]).sum() / weight)
            volume = float(group["volume"].sum())

            row: Dict[str, object] = {
                "symbol": symbol,
                "date": date,
                "sentiment_score": np.clip(score, -1.0, 1.0),
                "sentiment_magnitude": np.clip(magnitude, 0.0, 1.0),
                "sentiment_volume": volume,
            }

            if embedding_dim > 0 and "embedding" in group.columns:
                emb_vectors = [vec for vec in group["embedding"] if isinstance(vec, np.ndarray)]
                if emb_vectors:
                    stacked = np.vstack(
                        [
                            vec
                            if vec.size == embedding_dim
                            else np.pad(vec, (0, embedding_dim - vec.size))
                            for vec in emb_vectors
                        ]
                    )
                    emb_avg = stacked.mean(axis=0)
                else:
                    emb_avg = np.zeros(embedding_dim, dtype=np.float32)
                for idx in range(embedding_dim):
                    row[f"sentiment_emb_{idx:03d}"] = float(emb_avg[idx])

            aggregation_rows.append(row)

        if not aggregation_rows:
            return SentimentFeatureSet(frames={}, columns=[], embedding_dim=embedding_dim)

        agg_df = pd.DataFrame(aggregation_rows).sort_values("date")
        frames: Dict[str, pd.DataFrame] = {}
        for symbol, symbol_df in agg_df.groupby("symbol"):
            frame = symbol_df.drop(columns=["symbol"]).set_index("date").sort_index()
            frame = frame.ffill().fillna(0.0)
            frame = self._augment_multiscale_features(frame)
            frames[symbol] = frame

        feature_columns = list(next(iter(frames.values())).columns) if frames else []

        return SentimentFeatureSet(
            frames=frames, columns=feature_columns, embedding_dim=embedding_dim
        )

    def _augment_multiscale_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        base_cols = ["sentiment_score", "sentiment_magnitude", "sentiment_volume"]
        for base in base_cols:
            if base not in frame.columns:
                continue
            series = frame[base]
            for window in self.ema_windows:
                ema_col = f"{base}_ema_{window}"
                roc_col = f"{base}_roc_{window}"
                frame[ema_col] = series.ewm(span=window, adjust=False, min_periods=1).mean()
                pct = series.pct_change(window).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                frame[roc_col] = np.clip(pct, -5.0, 5.0)
        # 为情绪 embedding 生成能量指标
        if "sentiment_score" in frame.columns:
            frame["sentiment_score_energy"] = (
                frame["sentiment_score"].pow(2).rolling(5, min_periods=1).mean().fillna(0.0)
            )
        return frame


def list_sentiment_datasets(base_dir: Optional[str | Path] = None) -> List[Path]:
    if base_dir:
        base = Path(base_dir)
    else:
        base = DATA_ROOT / "news" / "data" / "exports"
    if not base.exists():
        return []
    results: List[Path] = []
    for pattern in ("news_fomalhaut_*.jsonl", "news_deepseek_*.jsonl"):
        for path in base.glob(pattern):
            if "errors" in path.stem.lower():
                continue
            results.append(path)
    return sorted(results)


def _parse_cutoff_time(value: dt_time | str | None) -> dt_time:
    """Parse cutoff time in 'HH:MM[:SS]' format; default to 09:30."""
    if value is None:
        return dt_time(9, 30)
    if isinstance(value, dt_time):
        return value

    text = str(value).strip()
    if not text:
        return dt_time(9, 30)

    parts = text.split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) >= 2 else 0
        second = int(parts[2]) if len(parts) >= 3 else 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
            raise ValueError("time component out of range")
        return dt_time(hour, minute, second)
    except (TypeError, ValueError, IndexError):
        return dt_time(9, 30)
