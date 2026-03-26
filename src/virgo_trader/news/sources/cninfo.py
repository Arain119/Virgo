"""Cninfo announcements source adapter."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from .. import config
from ..models import NewsItem, Target
from ..utils import SHANGHAI_TZ, build_headers, request_with_retry
from .base import NewsSource

logger = logging.getLogger(__name__)


class CninfoAnnouncementSource(NewsSource):
    """Official announcements from CNINFO with automatic window slicing."""

    BASE_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"

    def __init__(self, max_pages_per_window: int | None = 200, window_days: int = 120) -> None:
        super().__init__("cninfo", "announcement")
        self.max_pages_per_window = max_pages_per_window if max_pages_per_window else None
        self.window_days = max(30, window_days)
        self.session.headers.update(
            build_headers({"Content-Type": "application/json;charset=UTF-8"})
        )

    def fetch(self, target: Target, start_time: datetime, end_time: datetime) -> List[NewsItem]:
        records: List[NewsItem] = []
        column = "sse" if target.exchange == "SH" else "szse"
        windows = self._build_windows(start_time, end_time)
        for window_start_utc, window_end_utc, window_start_str, window_end_str in windows:
            page = 1
            while True:
                payload = {
                    "stock": target.cninfo_code,
                    "tabName": "fulltext",
                    "pageSize": config.CNINFO_PAGE_SIZE,
                    "pageNum": page,
                    "column": column,
                    "category": "",
                    "seDate": f"{window_start_str}~{window_end_str}",
                    "searchkey": "",
                }
                response = request_with_retry(self.session, "POST", self.BASE_URL, json=payload)
                data = response.json()
                announcements = data.get("announcements") or []
                if not announcements:
                    break

                reached_earliest = False
                for item in announcements:
                    timestamp = item.get("announcementTime")
                    if not timestamp:
                        continue
                    published = datetime.fromtimestamp(int(timestamp) / 1000, tz=timezone.utc)
                    if published < window_start_utc:
                        reached_earliest = True
                        break
                    if published > window_end_utc or published < start_time or published > end_time:
                        continue
                    url = f"https://static.cninfo.com.cn/{item.get('adjunctUrl')}"
                    records.append(
                        NewsItem(
                            source=self.name,
                            news_type=self.news_type,
                            target=target,
                            headline=item.get("announcementTitle", ""),
                            summary=item.get("announcementTitle"),
                            url=url,
                            published_at=published,
                            metadata={
                                "announcementId": item.get("announcementId"),
                                "adjunctType": item.get("adjunctType"),
                                "announcementTypeName": item.get("announcementTypeName"),
                                "column": column,
                            },
                        )
                    )
                if reached_earliest:
                    break
                page += 1
                if self.max_pages_per_window and page > self.max_pages_per_window:
                    break
        return records

    def _build_windows(
        self, start_time: datetime, end_time: datetime
    ) -> List[Tuple[datetime, datetime, str, str]]:
        local_start = start_time.astimezone(SHANGHAI_TZ).date()
        local_end = end_time.astimezone(SHANGHAI_TZ).date()
        windows: List[Tuple[datetime, datetime, str, str]] = []
        cursor = local_start
        while cursor <= local_end:
            window_end_date = min(cursor + timedelta(days=self.window_days - 1), local_end)
            start_dt = datetime.combine(cursor, datetime.min.time(), tzinfo=SHANGHAI_TZ).astimezone(
                timezone.utc
            )
            end_dt = datetime.combine(
                window_end_date, datetime.max.time().replace(microsecond=0), tzinfo=SHANGHAI_TZ
            ).astimezone(timezone.utc)
            windows.append((start_dt, end_dt, cursor.isoformat(), window_end_date.isoformat()))
            cursor = window_end_date + timedelta(days=1)
        return windows
