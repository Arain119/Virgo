"""Sina corporate announcements source adapter."""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone
from typing import List, Tuple

from ..models import NewsItem, Target
from ..utils import SHANGHAI_TZ, request_with_retry
from .base import NewsSource

logger = logging.getLogger(__name__)


class SinaCorpNewsSource(NewsSource):
    """Scrape Sina finance per-stock news archive pages."""

    BASE_URL = "http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php"
    ENTRY_PATTERN = re.compile(
        r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}).*?<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        re.S,
    )

    def __init__(self, max_pages: int = 80) -> None:
        super().__init__("sina_corp", "news")
        self.max_pages = max(1, max_pages)

    def fetch(self, target: Target, start_time: datetime, end_time: datetime) -> List[NewsItem]:
        symbol = target.symbol_alpha.lower()
        records: List[NewsItem] = []
        seen: set[Tuple[str, str]] = set()

        for page in range(1, self.max_pages + 1):
            params = {"symbol": symbol, "Page": page}
            try:
                response = request_with_retry(self.session, "GET", self.BASE_URL, params=params)
            except Exception as exc:  # pragma: no cover - network issues
                logger.warning("Sina corp news fetch failed for %s page %s: %s", symbol, page, exc)
                break
            response.encoding = "gbk"
            content = html.unescape(response.text).replace("\xa0", " ")
            matches = self.ENTRY_PATTERN.findall(content)
            if not matches:
                break

            reached_earliest = False
            for date_str, time_str, url, headline in matches:
                try:
                    local_dt = datetime.strptime(
                        f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                    ).replace(tzinfo=SHANGHAI_TZ)
                except ValueError:
                    continue
                published = local_dt.astimezone(timezone.utc)
                if published > end_time:
                    continue
                if published < start_time:
                    reached_earliest = True
                    continue
                title = headline.strip()
                key = (title, url)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    NewsItem(
                        source=self.name,
                        news_type=self.news_type,
                        target=target,
                        headline=title,
                        summary=None,
                        url=url,
                        published_at=published,
                        metadata={"page": page},
                    )
                )
            if reached_earliest:
                break

        return records
