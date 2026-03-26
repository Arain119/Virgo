"""Target universe builder for the news crawler.

Downloads (and caches) SSE 50 constituents and defines additional targets such
as the SSE50 ETF and composite index for crawling.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests

from . import config
from .models import Target
from .utils import build_headers, request_with_retry

logger = logging.getLogger(__name__)

CACHE_FILE = config.CACHE_DIR / f"{config.SSE_INDEX_CODE}_constituents.json"
CACHE_TTL = timedelta(hours=12)
CSINDEX_URL = (
    "https://oss-ch.csindex.com.cn/static/html/csindex/public/"
    f"uploads/file/autofile/cons/{config.SSE_INDEX_CODE}cons.xls"
)


def _is_cache_fresh(path: Path) -> bool:
    return (
        path.exists() and datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) < CACHE_TTL
    )


def _load_cache() -> List[dict]:
    if not _is_cache_fresh(CACHE_FILE):
        return []
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - cache corruption
        logger.warning("constituent cache corrupted, ignoring %s", CACHE_FILE)
        return []


def _save_cache(records: Iterable[dict]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(list(records), ensure_ascii=False, indent=2), encoding="utf-8")


def _download_constituents() -> List[dict]:
    session = requests.Session()
    response = request_with_retry(session, "GET", CSINDEX_URL, headers=build_headers())
    df = pd.read_excel(BytesIO(response.content), engine="xlrd")
    code_col = next(col for col in df.columns if "Constituent Code" in col)
    name_col = next(col for col in df.columns if "Constituent Name" in col and "Eng" not in col)
    exchange_col = next(col for col in df.columns if "Exchange" in col and "Eng" not in col)
    records: List[dict] = []
    for _, row in df.iterrows():
        code = str(row[code_col]).zfill(6)
        name = str(row[name_col]).strip()
        exchange_raw = str(row[exchange_col])
        exchange = "SH" if "上海" in exchange_raw or "SH" in exchange_raw.upper() else "SZ"
        records.append({"code": code, "name": name, "exchange": exchange})
    return records


def get_targets(force_refresh: bool = False) -> List[Target]:
    if not force_refresh:
        cached = _load_cache()
    else:
        cached = []

    if not cached:
        logger.info("Downloading latest SSE 50 constituents from CSIndex")
        cached = _download_constituents()
        _save_cache(cached)

    targets = [Target(**record) for record in cached]
    targets.append(
        Target(
            code=config.SSE_ETF_CODE,
            name=config.SSE_ETF_NAME,
            exchange="SH",
            asset_type="etf",
            keywords=[
                config.SSE_ETF_NAME,
                "上证50 ETF",
                config.SSE_ETF_CODE,
                f"SH{config.SSE_ETF_CODE}",
                "上证50基金",
            ],
        )
    )
    targets.append(
        Target(
            code=config.SH_COMPOSITE_INDEX_CODE,
            name=config.SH_COMPOSITE_INDEX_NAME,
            exchange="SH",
            asset_type="index",
            keywords=[
                config.SH_COMPOSITE_INDEX_NAME,
                "上证综指",
                "沪指",
                "上海综合指数",
                config.SH_COMPOSITE_INDEX_CODE,
                f"SH{config.SH_COMPOSITE_INDEX_CODE}",
            ],
        )
    )
    return targets
