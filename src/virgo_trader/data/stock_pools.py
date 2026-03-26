"""Definitions and helpers for stock pools / universes.

Provides a small registry (e.g. SSE50) and utilities to fetch constituent codes
via optional `akshare`, with a stable fallback list when the dependency is absent.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

try:
    import akshare as ak  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    ak = None


SSE50_FALLBACK_CODES: Tuple[str, ...] = (
    "600028.SH",
    "600030.SH",
    "600031.SH",
    "600036.SH",
    "600048.SH",
    "600050.SH",
    "600150.SH",
    "600276.SH",
    "600309.SH",
    "600406.SH",
    "600519.SH",
    "600690.SH",
    "600760.SH",
    "600809.SH",
    "600887.SH",
    "600900.SH",
    "600941.SH",
    "601012.SH",
    "601088.SH",
    "601127.SH",
    "601166.SH",
    "601211.SH",
    "601225.SH",
    "601288.SH",
    "601318.SH",
    "601328.SH",
    "601398.SH",
    "601600.SH",
    "601601.SH",
    "601628.SH",
    "601658.SH",
    "601668.SH",
    "601728.SH",
    "601766.SH",
    "601816.SH",
    "601857.SH",
    "601888.SH",
    "601899.SH",
    "601919.SH",
    "601985.SH",
    "601988.SH",
    "603259.SH",
    "603501.SH",
    "603993.SH",
    "688008.SH",
    "688012.SH",
    "688041.SH",
    "688111.SH",
    "688256.SH",
    "688981.SH",
)

SSE50_EXTRA_CODES: Tuple[str, ...] = ("510050.SH",)


STOCK_POOLS: Dict[str, Dict[str, str]] = {
    "sse50": {
        "label": "上证50股票池",
        "description": "自动同步上证50指数（000016.SH）最新成分股，供多标的训练使用。",
        "index_symbol": "000016",
        "fallback_codes": SSE50_FALLBACK_CODES,
        "extra_keywords": SSE50_EXTRA_CODES,
    }
}


def _format_code(value: str) -> str:
    code = value.strip().upper()
    if not code:
        return code
    if code.endswith((".SH", ".SZ", ".BJ")):
        return code
    # STAR Market代码和以6开头的沪市代码默认添加.SH
    if code.startswith(("6", "688")):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    return code


def _fetch_codes_via_akshare(index_symbol: str) -> List[str]:
    if ak is None:
        return []
    try:
        df = ak.index_stock_cons(symbol=index_symbol)
        if df is None or df.empty:
            return []
        column = None
        for candidate in ("品种代码", "证券代码", "stock_code"):
            if candidate in df.columns:
                column = candidate
                break
        if column is None:
            # Fallback to the first column if schema changes
            column = df.columns[0]
        codes = [
            _format_code(str(code).zfill(6))
            for code in df[column].astype(str).tolist()
            if str(code).strip()
        ]
        return codes
    except Exception as exc:  # pragma: no cover - network failures handled gracefully
        logging.warning("无法获取指数 %s 成分股：%s。", index_symbol, exc)
        return []


def list_stock_pools() -> List[Dict[str, str]]:
    """Return metadata of all registered stock pools for UI consumption."""
    return [
        {"key": key, "label": meta["label"], "description": meta.get("description", "")}
        for key, meta in STOCK_POOLS.items()
    ]


@lru_cache(maxsize=None)
def get_stock_pool_codes(pool_key: str, use_live: bool = True) -> Tuple[str, ...]:
    """Return an immutable tuple of stock codes for the given pool."""
    if not pool_key:
        raise KeyError("pool_key is required.")
    normalized_key = pool_key.lower()
    pool = STOCK_POOLS.get(normalized_key)
    if not pool:
        raise KeyError(f"Unknown stock pool '{pool_key}'.")

    codes: Iterable[str] = ()
    if use_live and pool.get("index_symbol"):
        codes = _fetch_codes_via_akshare(pool["index_symbol"])

    if not codes:
        codes = pool["fallback_codes"]

    extra_keywords: Iterable[str] = pool.get("extra_keywords", ())
    ordered_codes = list(extra_keywords) + list(codes)

    # Remove duplicates while preserving order
    unique_codes = tuple(dict.fromkeys(_format_code(code) for code in ordered_codes if code))
    if not unique_codes:
        raise ValueError(f"No valid codes resolved for stock pool '{pool_key}'.")
    return unique_codes


def get_stock_pool_label(pool_key: str) -> str:
    pool = STOCK_POOLS.get(pool_key.lower())
    if not pool:
        raise KeyError(f"Unknown stock pool '{pool_key}'.")
    return pool["label"]
