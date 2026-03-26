"""Caixin news source adapter."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import akshare as ak

from ..models import NewsItem, Target
from ..utils import SHANGHAI_TZ, utc_now
from .base import NewsSource

logger = logging.getLogger(__name__)


class CaixinNewsSource(NewsSource):
    """Caixin global / 财新要闻（akshare.stock_news_main_cx），可追溯至 2020 年。"""

    def __init__(
        self,
        refresh_interval: timedelta = timedelta(hours=6),
        max_per_symbol: int = 200,
    ) -> None:
        super().__init__("caixin_news", "news")
        self.refresh_interval = refresh_interval
        self.max_per_symbol = max_per_symbol
        self._cache: Optional[List[dict]] = None
        self._cached_at: Optional[datetime] = None

    def fetch(self, target: Target, start_time: datetime, end_time: datetime) -> List[NewsItem]:
        articles = self._load_articles()
        keywords = self._select_keywords(target)
        tokens = [kw.lower() for kw in keywords]
        collected: List[NewsItem] = []

        for article in articles:
            published = article["published_at"]
            if published < start_time or published > end_time:
                continue
            blob = article["text_blob"]
            if tokens and not any(token in blob for token in tokens):
                continue
            collected.append(
                NewsItem(
                    source=self.name,
                    news_type=self.news_type,
                    target=target,
                    headline=article["headline"],
                    summary=article["summary"],
                    url=article["url"],
                    published_at=published,
                    metadata={"provider": "caixin", "raw_pub_time": article["raw_time"]},
                )
            )
            if self.max_per_symbol and len(collected) >= self.max_per_symbol:
                break
        return collected

    def _load_articles(self) -> List[dict]:
        now = utc_now()
        if self._cache and self._cached_at and now - self._cached_at < self.refresh_interval:
            return self._cache
        try:
            df = ak.stock_news_main_cx()
        except Exception as exc:  # pragma: no cover - akshare runtime issue
            logger.warning("Caixin news fetch failed: %s", exc)
            return self._cache or []

        articles: List[dict] = []
        for _, row in df.iterrows():
            headline = str(row.get("tag") or "").strip()
            summary = str(row.get("summary") or "").strip()
            url = str(row.get("url") or "").strip()
            pub_time = str(row.get("pub_time") or "").strip()
            if not headline or not pub_time:
                continue
            try:
                published_local = datetime.strptime(pub_time, "%Y-%m-%d %H:%M:%S.%f").replace(
                    tzinfo=SHANGHAI_TZ
                )
            except ValueError:
                try:
                    published_local = datetime.strptime(pub_time, "%Y-%m-%d %H:%M:%S").replace(
                        tzinfo=SHANGHAI_TZ
                    )
                except ValueError:
                    continue
            articles.append(
                {
                    "headline": headline,
                    "summary": summary or headline,
                    "url": url,
                    "raw_time": pub_time,
                    "published_at": published_local.astimezone(timezone.utc),
                    "text_blob": f"{headline} {summary}".lower(),
                }
            )
        articles.sort(key=lambda item: item["published_at"], reverse=True)
        self._cache = articles
        self._cached_at = now
        return articles

    def _select_keywords(self, target: Target) -> List[str]:
        base = [
            target.name,
            target.code,
            target.symbol_alpha,
        ]
        base.extend(target.keywords or [])
        seen = set()
        tokens: List[str] = []
        for keyword in base:
            token = (keyword or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens
