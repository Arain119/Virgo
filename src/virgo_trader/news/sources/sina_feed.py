"""Sina rolling feed news source adapter."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Sequence

from ..models import NewsItem, Target
from ..utils import request_with_retry
from .base import NewsSource

logger = logging.getLogger(__name__)


class SinaFeedSource(NewsSource):
    """Sina rolling news feed filtered by keywords."""

    BASE_URL = "https://feed.mix.sina.com.cn/api/roll/get"
    DEFAULT_LIDS: Sequence[int] = (
        2509,  # 公司新闻
        2510,  # 证券要闻
        2511,  # 研报资讯
        2512,  # 宏观新闻
        2515,  # 行业热点
        2516,  # 财经国内
    )

    def __init__(
        self,
        lids: Sequence[int] | None = None,
        pages_per_lid: int = 5,
        page_size: int = 50,
    ) -> None:
        super().__init__("sina_feed", "news")
        self.lids = tuple(lids) if lids else self.DEFAULT_LIDS
        self.pages_per_lid = max(1, pages_per_lid)
        self.page_size = max(10, page_size)

    def fetch(self, target: Target, start_time: datetime, end_time: datetime) -> List[NewsItem]:
        keywords = self._select_keywords(target)
        keyword_tokens = [kw.lower() for kw in keywords]
        results: List[NewsItem] = []
        dedupe = set()

        for lid in self.lids:
            for page in range(1, self.pages_per_lid + 1):
                params = {
                    "pageid": 153,
                    "lid": lid,
                    "num": self.page_size,
                    "page": page,
                    "version": 2,
                }
                resp = request_with_retry(self.session, "GET", self.BASE_URL, params=params)
                data = resp.json().get("result", {}).get("data") or []
                if not data:
                    break
                reached_earliest = False
                for item in data:
                    try:
                        published = datetime.fromtimestamp(
                            int(item.get("ctime", 0)), tz=timezone.utc
                        )
                    except (TypeError, ValueError):
                        continue
                    if published < start_time:
                        reached_earliest = True
                        break
                    if published > end_time:
                        continue
                    blob = " ".join(
                        filter(
                            None,
                            [
                                item.get("title"),
                                item.get("summary"),
                                item.get("intro"),
                                item.get("wapsummary"),
                                item.get("keywords"),
                            ],
                        )
                    ).lower()
                    if blob and not any(token in blob for token in keyword_tokens):
                        continue
                    headline = (item.get("title") or "").strip()
                    if not headline:
                        continue
                    url = item.get("url") or item.get("wapurl") or ""
                    dedupe_key = (headline, url)
                    if dedupe_key in dedupe:
                        continue
                    dedupe.add(dedupe_key)
                    summary = (
                        item.get("summary")
                        or item.get("wapsummary")
                        or item.get("intro")
                        or headline
                    )
                    results.append(
                        NewsItem(
                            source=self.name,
                            news_type=self.news_type,
                            target=target,
                            headline=headline,
                            summary=summary,
                            url=url,
                            published_at=published,
                            metadata={
                                "lid": lid,
                                "page": page,
                                "docid": item.get("docid"),
                                "media_name": item.get("media_name"),
                            },
                        )
                    )
                if reached_earliest:
                    break
        return results

    def _select_keywords(self, target: Target) -> List[str]:
        base = [
            target.name,
            target.code,
            target.symbol_alpha,
        ]
        base.extend(target.keywords or [])

        seen = set()
        result: List[str] = []
        for keyword in base:
            if not keyword:
                continue
            token = keyword.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result or [target.symbol_alpha]
