"""News source adapters.

This package provides site-specific implementations that fetch news or
announcements and normalize them into `NewsItem` objects.
"""

from .caixin import CaixinNewsSource
from .cninfo import CninfoAnnouncementSource
from .eastmoney import EastmoneyNewsSource
from .sina import SinaNewsSource
from .sina_corp import SinaCorpNewsSource
from .sina_feed import SinaFeedSource
from .sse import SSEAnnouncementSource

__all__ = [
    "SinaNewsSource",
    "SinaFeedSource",
    "SinaCorpNewsSource",
    "EastmoneyNewsSource",
    "CaixinNewsSource",
    "CninfoAnnouncementSource",
    "SSEAnnouncementSource",
]
