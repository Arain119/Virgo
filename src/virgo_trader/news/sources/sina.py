"""Sina search-based news source adapter."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List

from bs4 import BeautifulSoup

from ..models import NewsItem, Target
from ..utils import SHANGHAI_TZ, request_with_retry, utc_now
from .base import NewsSource

logger = logging.getLogger(__name__)


class SinaNewsSource(NewsSource):
    """Use search.sina.com.cn to pull keyword news, paging until达到起始时间."""

    SEARCH_URL = "https://search.sina.com.cn/"
    TIME_PATTERN = re.compile(
        r"(\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}|\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}|"
        r"\d{4}-\d{1,2}-\d{1,2}\s*\d{1,2}:\d{2}|\d{4}-\d{1,2}-\d{1,2}|"
        r"\d{1,2}月\d{1,2}日|\d+分钟前|\d+小时前|\d+天前)"
    )

    def __init__(self, keywords_per_target: int = 3, pages_per_keyword: int = 12) -> None:
        super().__init__("sina_search", "news")
        self.session.headers.update({"Referer": self.SEARCH_URL})
        self.keywords_per_target = max(1, keywords_per_target)
        self.pages_per_keyword = max(1, pages_per_keyword)

    def fetch(self, target: Target, start_time: datetime, end_time: datetime) -> List[NewsItem]:
        keywords = self._select_keywords(target)
        records: List[NewsItem] = []
        dedupe = set()
        for keyword in keywords:
            stop_keyword = False
            for page in range(1, self.pages_per_keyword + 1):
                params = {
                    "q": keyword,
                    "c": "news",
                    "range": "all",
                    "num": 20,
                    "page": page,
                    "col": "1_7",
                    "source": "",
                }
                response = request_with_retry(self.session, "GET", self.SEARCH_URL, params=params)
                response.encoding = "utf-8"
                soup = BeautifulSoup(response.text, "html.parser")
                items = soup.select("div.box-result")
                if not items:
                    break
                for node in items:
                    title_node = node.select_one("h2 a")
                    if not title_node:
                        continue
                    headline = (title_node.get_text(strip=True) or "").replace("\n", " ")
                    url = title_node.get("href", "")
                    summary_node = node.select_one("p.content")
                    summary = summary_node.get_text(" ", strip=True) if summary_node else None
                    meta_node = node.select_one("span.fgray_time")
                    source_name, published = self._parse_meta(meta_node)
                    if not published:
                        continue
                    if published < start_time:
                        stop_keyword = True
                        break
                    if published > end_time:
                        continue
                    dedupe_key = (headline, url)
                    if dedupe_key in dedupe:
                        continue
                    dedupe.add(dedupe_key)
                    records.append(
                        NewsItem(
                            source=self.name,
                            news_type=self.news_type,
                            target=target,
                            headline=headline,
                            summary=summary,
                            url=url,
                            published_at=published,
                            metadata={
                                "keyword": keyword,
                                "source_name": source_name,
                                "page": page,
                            },
                        )
                    )
                if stop_keyword:
                    break
        return records

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
            if len(result) >= self.keywords_per_target:
                break
        return result

    def _parse_meta(self, meta_node) -> tuple[str | None, datetime | None]:
        if not meta_node:
            return None, None
        text = meta_node.get_text(" ", strip=True)
        time_match = self.TIME_PATTERN.search(text)
        time_str = time_match.group(1) if time_match else ""
        source_name = text.replace(time_str, "").strip() if time_str else text
        published = self._parse_time_str(time_str)
        return source_name, published

    def _parse_time_str(self, value: str) -> datetime | None:
        if not value:
            return None
        value = value.strip()
        now_local = utc_now().astimezone(SHANGHAI_TZ)

        relative_match = re.match(r"(\d+)(分钟前|小时前|天前)", value)
        if relative_match:
            amount = int(relative_match.group(1))
            unit = relative_match.group(2)
            delta_map = {
                "分钟前": timedelta(minutes=amount),
                "小时前": timedelta(hours=amount),
                "天前": timedelta(days=amount),
            }
            dt = now_local - delta_map.get(unit, timedelta())
            return dt.astimezone(timezone.utc)

        normalized = (
            value.replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-").strip()
        )
        if " " not in normalized:
            pattern = "%m-%d"
            if len(normalized.split("-")) == 3:
                pattern = "%Y-%m-%d"
            try:
                dt = datetime.strptime(normalized, pattern)
                if pattern == "%m-%d":
                    dt = dt.replace(year=now_local.year)
                return dt.replace(tzinfo=SHANGHAI_TZ).astimezone(timezone.utc)
            except ValueError:
                return None
        else:
            parts = normalized.split()
            date_part = parts[0]
            time_part = parts[1]
            if len(date_part.split("-")) == 2:
                date_part = f"{now_local.year}-{date_part}"
            try:
                dt = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M")
                return dt.replace(tzinfo=SHANGHAI_TZ).astimezone(timezone.utc)
            except ValueError:
                return None
