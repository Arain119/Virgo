"""Data models for the `news` crawler package.

Defines `Target` (securities/keywords) and `NewsItem` (normalized crawler output)
used across sources, repository, and export pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import utils


@dataclass(slots=True)
class Target:
    code: str
    name: str
    exchange: str  # SH or SZ
    asset_type: str = "equity"
    keywords: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.exchange = self.exchange.upper()
        if self.keywords is None:
            self.keywords = self.default_keywords()

    def default_keywords(self) -> List[str]:
        base = {self.name, self.code, self.symbol_numeric, self.symbol_alpha}
        base.add(self.name.replace("股份", "").replace("有限公司", ""))
        return sorted({kw for kw in base if kw})

    @property
    def symbol_alpha(self) -> str:
        return f"{self.exchange}{self.code}"

    @property
    def symbol_numeric(self) -> str:
        return self.code

    @property
    def cninfo_code(self) -> str:
        return f"{self.exchange.lower()}{self.code}"


@dataclass(slots=True)
class NewsItem:
    source: str
    news_type: str
    target: Target
    headline: str
    url: str
    published_at: datetime
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        utc_time = utils.ensure_tz(self.published_at)
        return {
            "source": self.source,
            "news_type": self.news_type,
            "symbol": self.target.symbol_alpha,
            "security_name": self.target.name,
            "headline": utils.sanitize(self.headline),
            "summary": utils.sanitize(self.summary),
            "url": self.url,
            "published_at": utc_time.isoformat(),
            "ingested_at": utils.utc_now().isoformat(),
            "keywords": utils.dumps(self.keywords or self.target.keywords),
            "extra": utils.dumps(self.metadata),
            "hash": utils.make_digest(
                self.source, self.target.symbol_alpha, self.headline, self.url
            ),
        }
