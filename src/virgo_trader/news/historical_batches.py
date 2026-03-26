"""Historical batch crawler for the news dataset.

Fetches news/announcements over configured date ranges with per-day limits,
persists results into SQLite, and writes range-scoped CSV exports.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from . import config, targets
from .models import NewsItem, Target
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
from .utils import SHANGHAI_TZ, utc_now

logger = logging.getLogger(__name__)

DEFAULT_RANGES = [
    ("2022-01-01", "2024-12-31", "2022_2024"),
    ("2024-01-01", "2025-12-31", "2024_2025"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch crawler for SSE 50 constituents + ETF with per-day limits",
    )
    parser.add_argument(
        "--per-day-limit",
        type=int,
        default=5,
        help="max items per target per day (0 = unlimited)",
    )
    parser.add_argument(
        "--range",
        action="append",
        metavar="START:END[:LABEL]",
        help="custom ranges, e.g. 2022-01-01:2023-12-31:phase_a",
    )
    parser.add_argument(
        "--sina-pages",
        type=int,
        default=8,
        help="Sina search pages per keyword (default 8)",
    )
    parser.add_argument(
        "--sina-keywords",
        type=int,
        default=4,
        help="keywords per target for Sina search (default 4)",
    )
    parser.add_argument(
        "--feed-pages",
        type=int,
        default=4,
        help="pages per Sina feed channel (default 4)",
    )
    parser.add_argument(
        "--sina-corp-pages",
        type=int,
        default=80,
        help="maximum pages for Sina corp archive (default 80)",
    )
    parser.add_argument(
        "--disable-feed",
        action="store_true",
        help="skip Sina feed source",
    )
    parser.add_argument(
        "--enable-sina-search",
        action="store_true",
        help="enable Sina search source",
    )
    parser.add_argument(
        "--disable-caixin",
        action="store_true",
        help="skip Caixin news source",
    )
    parser.add_argument(
        "--disable-eastmoney",
        action="store_true",
        help="skip Eastmoney news source",
    )
    parser.add_argument(
        "--disable-sina-corp",
        action="store_true",
        help="skip Sina corp archive source",
    )
    parser.add_argument(
        "--disable-cninfo",
        action="store_true",
        help="skip Cninfo announcements",
    )
    parser.add_argument(
        "--disable-sse",
        action="store_true",
        help="skip SSE announcements",
    )
    parser.add_argument(
        "--sse-page-size",
        type=int,
        default=50,
        help="SSE announcement page size (default 50)",
    )
    parser.add_argument(
        "--cninfo-window-days",
        type=int,
        default=120,
        help="Cninfo window size in days (default 120)",
    )
    parser.add_argument(
        "--cninfo-pages",
        type=int,
        default=200,
        help="max pages per Cninfo window (0 = unlimited)",
    )
    parser.add_argument(
        "--limit-targets",
        type=int,
        help="limit number of targets for testing",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="filter to specific symbols (e.g. SH600028 600036)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level",
    )
    return parser.parse_args()


def ensure_ranges(args: argparse.Namespace) -> List[Tuple[datetime, datetime, str]]:
    raw_ranges = args.range or [":".join(r) for r in DEFAULT_RANGES]
    resolved: List[Tuple[datetime, datetime, str]] = []
    now = utc_now()
    for raw in raw_ranges:
        parts = raw.split(":")
        if len(parts) < 2:
            raise ValueError(f"非法区间格式：{raw}，应为 START:END[:LABEL]")
        start_str, end_str = parts[0], parts[1]
        label = parts[2] if len(parts) > 2 else f"{start_str}_{end_str}"
        start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=SHANGHAI_TZ)
        end = datetime.strptime(end_str, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=SHANGHAI_TZ
        )
        if end.astimezone(timezone.utc) > now:
            end = now.astimezone(SHANGHAI_TZ)
        resolved.append((start.astimezone(timezone.utc), end.astimezone(timezone.utc), label))
    return resolved


class RangeWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = path.open("w", newline="", encoding="utf-8-sig")
        self.writer: csv.DictWriter | None = None

    def write(self, record: Dict[str, str]) -> None:
        if self.writer is None:
            fieldnames = list(record.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
            self.writer.writeheader()
        self.writer.writerow(record)

    def close(self) -> None:
        self.file.close()


def crawl_range(
    label: str,
    start: datetime,
    end: datetime,
    targets_list: Sequence[Target],
    per_day_limit: int,
    sources: Sequence,
    repo: NewsRepository,
) -> Path | None:
    logger.info("开始抓取区间 %s (%s 至 %s)", label, start, end)
    per_day_counters: Dict[Tuple[str, date], int] = defaultdict(int)
    dedupe_set = set()
    output_path = config.EXPORT_DIR / f"news_{label}.csv"
    writer = RangeWriter(output_path)
    accepted = 0

    def process_items(items: List[NewsItem]) -> None:
        nonlocal accepted
        if not items:
            return
        for item in sorted(items, key=lambda x: x.published_at, reverse=True):
            dedupe_key = (item.source, item.target.symbol_alpha, item.headline.strip(), item.url)
            if dedupe_key in dedupe_set:
                continue
            dedupe_set.add(dedupe_key)
            if per_day_limit > 0:
                local_day = item.published_at.astimezone(SHANGHAI_TZ).date()
                counter_key = (item.target.symbol_alpha, local_day)
                if per_day_counters[counter_key] >= per_day_limit:
                    continue
                per_day_counters[counter_key] += 1
            record = item.to_record()
            record["range_label"] = label
            writer.write(record)
            repo.insert_records([record])
            accepted += 1

    try:
        for target in targets_list:
            for source in sources:
                try:
                    fetched = source.fetch(target, start, end)
                except Exception as exc:  # pragma: no cover - 网络波动
                    logger.exception(
                        "抓取 %s from %s 失败：%s", target.symbol_alpha, source.name, exc
                    )
                    continue
                process_items(fetched)
    finally:
        writer.close()

    if accepted == 0:
        logger.warning("区间 %s 暂无可用数据，已移除空文件", label)
        output_path.unlink(missing_ok=True)
        return None
    logger.info("区间 %s 完成，保留 %s 条 -> %s", label, accepted, output_path)
    return output_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    ranges = ensure_ranges(args)
    targets_list = targets.get_targets()
    if args.symbols:
        normalized: set[str] = set()
        for raw in args.symbols:
            token = (raw or "").strip().upper()
            if not token:
                continue
            normalized.add(token)
            if token.startswith(("SH", "SZ")):
                normalized.add(token[2:])
        filtered = [
            target
            for target in targets_list
            if target.symbol_alpha.upper() in normalized or target.code in normalized
        ]
        if not filtered:
            raise SystemExit("No targets matched --symbols filters")
        targets_list = filtered
    if args.limit_targets:
        targets_list = targets_list[: args.limit_targets]

    repo = NewsRepository()
    sources = []
    if args.enable_sina_search:
        sources.append(
            SinaNewsSource(
                pages_per_keyword=args.sina_pages,
                keywords_per_target=args.sina_keywords,
            )
        )
    if not args.disable_eastmoney:
        sources.append(EastmoneyNewsSource())
    if not args.disable_feed:
        sources.append(SinaFeedSource(pages_per_lid=args.feed_pages))
    if not args.disable_sina_corp:
        sources.append(SinaCorpNewsSource(max_pages=args.sina_corp_pages))
    if not args.disable_caixin:
        sources.append(CaixinNewsSource())
    if not args.disable_cninfo:
        sources.append(
            CninfoAnnouncementSource(
                max_pages_per_window=None
                if args.cninfo_pages is not None and args.cninfo_pages <= 0
                else args.cninfo_pages,
                window_days=args.cninfo_window_days,
            )
        )
    if not args.disable_sse:
        sources.append(SSEAnnouncementSource(page_size=args.sse_page_size))

    try:
        for start, end, label in ranges:
            crawl_range(
                label=label,
                start=start,
                end=end,
                targets_list=targets_list,
                per_day_limit=args.per_day_limit,
                sources=sources,
                repo=repo,
            )
    finally:
        for source in sources:
            source.close()


if __name__ == "__main__":
    main()
