"""Eastmoney news source adapter."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from math import ceil
from typing import List

from ..models import NewsItem, Target
from ..utils import SHANGHAI_TZ, request_with_retry
from .base import NewsSource

logger = logging.getLogger(__name__)


class EastmoneyNewsSource(NewsSource):
    """Fetch Eastmoney search results (search-api-web) for each target keyword."""

    API_URL = "https://search-api-web.eastmoney.com/search/jsonp"
    CALLBACK = "jQuery3510875346244069884_1668256937995"
    REFERER = "http://so.eastmoney.com"

    def __init__(self, max_records: int = 1000, keywords_per_target: int = 3) -> None:
        super().__init__("eastmoney_news", "news")
        self.max_records = max(20, max_records)
        self.keywords_per_target = max(1, keywords_per_target)

    def fetch(self, target: Target, start_time: datetime, end_time: datetime) -> List[NewsItem]:
        records: List[NewsItem] = []
        keywords = self._select_keywords(target)
        dedupe = set()
        for keyword in keywords[: self.keywords_per_target]:
            fetched = 0
            page_size = min(50, self.max_records)
            max_pages = max(1, ceil(self.max_records / page_size))
            stop_keyword = False
            for page in range(1, max_pages + 1):
                articles = self._fetch_articles(keyword, page, page_size)
                if not articles:
                    break
                for item in articles:
                    headline = self._clean_text(item.get("title"))
                    summary = self._clean_text(item.get("content") or "")
                    published_raw = item.get("date") or ""
                    url_code = item.get("code") or ""
                    url = (
                        f"http://finance.eastmoney.com/a/{url_code}.html"
                        if url_code
                        else item.get("url", "")
                    )
                    if not headline or not published_raw:
                        continue
                    try:
                        dt_local = datetime.strptime(published_raw, "%Y-%m-%d %H:%M:%S").replace(
                            tzinfo=SHANGHAI_TZ
                        )
                        published = dt_local.astimezone(timezone.utc)
                    except ValueError:
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
                            summary=summary or headline,
                            url=url,
                            published_at=published,
                            metadata={
                                "provider": "eastmoney",
                                "raw_time": published_raw,
                                "keyword": keyword,
                                "source": item.get("mediaName"),
                            },
                        )
                    )
                    fetched += 1
                    if fetched >= self.max_records:
                        break
                if fetched >= self.max_records or stop_keyword:
                    break
        return records

    def _fetch_articles(self, keyword: str, page: int, page_size: int) -> List[dict]:
        payload = {
            "uid": "",
            "keyword": keyword,
            "type": ["cmsArticle"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticle": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": page,
                    "pageSize": page_size,
                    "preTag": "<em>",
                    "postTag": "</em>",
                }
            },
        }
        params = {
            "cb": self.CALLBACK,
            "param": json.dumps(payload, ensure_ascii=False),
        }
        try:
            resp = request_with_retry(
                self.session,
                "GET",
                self.API_URL,
                params=params,
                headers={"Referer": self.REFERER},
                allow_relax_tls=False,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Eastmoney request failed for %s page %s: %s", keyword, page, exc)
            return []
        text = resp.text.strip()
        prefix = f"{self.CALLBACK}("
        if text.startswith(prefix):
            text = text[len(prefix) : -1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Eastmoney JSON解析失败 %s page %s: %s", keyword, page, exc)
            return []
        result = data.get("result") or {}
        return result.get("cmsArticle") or []

    def _clean_text(self, text: str | None) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"<\/?em>", "", text)
        cleaned = cleaned.replace("\u3000", " ").replace("\r", " ").replace("\n", " ")
        return re.sub(r"\s+", " ", cleaned).strip()

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
