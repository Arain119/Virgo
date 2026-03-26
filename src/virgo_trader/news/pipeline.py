"""News crawling orchestration.

`NewsCrawler` coordinates multiple data sources, persists results to SQLite, and
optionally exports a recent window for labeling / downstream pipelines.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Dict, List, Sequence

from . import config, targets
from .models import Target
from .repository import NewsRepository
from .sources import (
    CaixinNewsSource,
    CninfoAnnouncementSource,
    EastmoneyNewsSource,
    SinaCorpNewsSource,
    SinaFeedSource,
    SinaNewsSource,
    SSEAnnouncementSource,
)
from .sources.base import NewsSource
from .state import StateStore
from .utils import utc_now

logger = logging.getLogger(__name__)


def _default_start_datetime() -> datetime:
    return datetime.combine(config.DEFAULT_START_DATE, time.min, tzinfo=timezone.utc)


class NewsCrawler:
    """Fetch news from multiple sources and persist them to the repository."""

    def __init__(
        self,
        news_sources: Sequence[NewsSource] | None = None,
        repository: NewsRepository | None = None,
        state_store: StateStore | None = None,
    ) -> None:
        self.sources: List[NewsSource] = list(
            news_sources
            or [
                SinaNewsSource(),
                SinaCorpNewsSource(),
                EastmoneyNewsSource(),
                SinaFeedSource(),
                CaixinNewsSource(),
                CninfoAnnouncementSource(),
                SSEAnnouncementSource(),
            ]
        )
        self.repo = repository or NewsRepository()
        self.state = state_store or StateStore()

    def run(
        self,
        mode: str = "incremental",
        start: datetime | None = None,
        end: datetime | None = None,
        limit_targets: int | None = None,
        export: bool = True,
    ) -> Dict[str, Dict[str, int]]:
        start_dt = start or _default_start_datetime()
        end_dt = end or utc_now()
        all_targets = targets.get_targets()
        if limit_targets:
            all_targets = all_targets[:limit_targets]

        metrics: Dict[str, Dict[str, int]] = {
            source.name: {"fetched": 0, "inserted": 0} for source in self.sources
        }

        sorted_targets = sorted(
            all_targets,
            key=lambda tgt: min(
                (
                    self.state.get(source.name, tgt.symbol_alpha) or start_dt
                    for source in self.sources
                ),
                default=start_dt,
            ),
        )

        try:
            for target in sorted_targets:
                for source in self.sources:
                    target_start = self._resolve_start(mode, source.name, target, start_dt)
                    try:
                        items = source.fetch(target, target_start, end_dt)
                    except Exception as exc:  # pragma: no cover - network/runtime issues
                        logger.exception(
                            "Failed to fetch %s from %s: %s", target.symbol_alpha, source.name, exc
                        )
                        continue
                    if not items:
                        continue
                    metrics[source.name]["fetched"] += len(items)
                    inserted = self.repo.insert_records(item.to_record() for item in items)
                    metrics[source.name]["inserted"] += inserted
                    latest_ts = max(item.published_at for item in items)
                    self.state.update(source.name, target.symbol_alpha, latest_ts)
        finally:
            self.state.persist()
            for source in self.sources:
                source.close()

        if export:
            self.repo.export_recent()
        return metrics

    def _resolve_start(
        self, mode: str, source_name: str, target: Target, default_start: datetime
    ) -> datetime:
        last_seen = self.state.get(source_name, target.symbol_alpha)
        if mode == "incremental" and last_seen:
            return max(last_seen, default_start)
        return default_start
