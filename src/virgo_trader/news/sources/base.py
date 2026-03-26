"""Base interfaces for news source implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import requests

from ..models import NewsItem, Target
from ..utils import build_headers


class NewsSource(ABC):
    def __init__(self, name: str, news_type: str) -> None:
        self.name = name
        self.news_type = news_type
        self.session = requests.Session()
        self.session.headers.update(build_headers())

    @abstractmethod
    def fetch(self, target: Target, start_time: datetime, end_time: datetime) -> List[NewsItem]: ...

    def close(self) -> None:
        self.session.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
