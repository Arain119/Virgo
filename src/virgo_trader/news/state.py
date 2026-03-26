"""Persistent crawl state storage.

Stores the latest crawl timestamp per (source, symbol) so incremental crawling
can resume deterministically across runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from . import config


class StateStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.STATE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Dict[str, str]] = self._load()

    def _load(self) -> Dict[str, Dict[str, str]]:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return {src: dict(entries) for src, entries in data.items()}
        return {}

    def get(self, source: str, symbol: str) -> Optional[datetime]:
        raw = self._state.get(source, {}).get(symbol)
        return datetime.fromisoformat(raw) if raw else None

    def update(self, source: str, symbol: str, value: datetime) -> None:
        self._state.setdefault(source, {})[symbol] = value.isoformat()

    def persist(self) -> None:
        self.path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
