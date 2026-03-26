"""Persistence layer for the `news` package (SQLite).

The crawler stores normalized news items into a single table and can export
recent records to CSV for downstream sentiment labeling / model training.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from . import config, utils


class NewsRepository:
    """SQLite repository for `news_items` records."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS news_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    news_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    security_name TEXT,
                    headline TEXT NOT NULL,
                    summary TEXT,
                    url TEXT,
                    published_at TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    keywords TEXT,
                    extra TEXT,
                    hash TEXT NOT NULL UNIQUE
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news_items(symbol, published_at)"
            )

    def insert_records(self, records: Iterable[Dict[str, str]]) -> int:
        inserted = 0
        sql = """
        INSERT INTO news_items
        (source, news_type, symbol, security_name, headline, summary,
         url, published_at, ingested_at, keywords, extra, hash)
        VALUES
        (:source, :news_type, :symbol, :security_name, :headline, :summary,
         :url, :published_at, :ingested_at, :keywords, :extra, :hash)
        ON CONFLICT(hash) DO UPDATE SET
            source=excluded.source,
            news_type=excluded.news_type,
            symbol=excluded.symbol,
            security_name=excluded.security_name,
            headline=excluded.headline,
            summary=excluded.summary,
            url=excluded.url,
            published_at=excluded.published_at,
            ingested_at=excluded.ingested_at,
            keywords=excluded.keywords,
            extra=excluded.extra
        """
        with self.conn:
            for record in records:
                cur = self.conn.execute(sql, record)
                inserted += cur.rowcount
        return inserted

    def export_recent(self, days: int = config.EXPORT_WINDOW_DAYS) -> Path:
        cutoff = (utils.utc_now() - timedelta(days=days)).isoformat()
        query = "SELECT * FROM news_items WHERE published_at >= ? ORDER BY published_at DESC"
        df = pd.read_sql_query(query, self.conn, params=[cutoff])
        config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.EXPORT_PATH, index=False, encoding="utf-8-sig")
        return config.EXPORT_PATH
