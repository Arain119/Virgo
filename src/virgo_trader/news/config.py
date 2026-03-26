"""Configuration for the `news` crawler package (paths, defaults, retry policy)."""

from __future__ import annotations

from datetime import date

from ..utils.paths import DATA_ROOT

# Keep crawler artifacts (SQLite, cache, exports, logs) outside the package tree.
DATA_BASE_DIR = DATA_ROOT / "news"

DATA_DIR = DATA_BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXPORT_DIR = DATA_DIR / "exports"
LOG_DIR = DATA_BASE_DIR / "logs"
STATE_DIR = DATA_BASE_DIR / "state"
CACHE_DIR = DATA_BASE_DIR / "cache"

DB_PATH = DATA_DIR / "news_items.db"
STATE_PATH = STATE_DIR / "crawler_state.json"
EXPORT_PATH = EXPORT_DIR / "latest_news.csv"

SSE_INDEX_CODE = "000016"
SSE_INDEX_NAME = "上证50指数"
SSE_ETF_CODE = "510050"
SSE_ETF_NAME = "上证50ETF"
SH_COMPOSITE_INDEX_CODE = "000001"
SH_COMPOSITE_INDEX_NAME = "上证指数"

DEFAULT_START_DATE = date(2020, 1, 1)
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
REQUEST_GAP_SECONDS = 1.0
MAX_PAGES_PER_KEYWORD = 5
CNINFO_PAGE_SIZE = 50
EXPORT_WINDOW_DAYS = 90

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
