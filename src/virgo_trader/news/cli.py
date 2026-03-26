"""Command-line interface for the news crawling pipeline.

Configures sources and date ranges, runs the crawler, and optionally exports a
consolidated CSV dataset.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from . import config
from .pipeline import NewsCrawler
from .sources import (
    CaixinNewsSource,
    CninfoAnnouncementSource,
    EastmoneyNewsSource,
    SinaCorpNewsSource,
    SinaFeedSource,
    SinaNewsSource,
    SSEAnnouncementSource,
)
from .utils import SHANGHAI_TZ


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated news crawler for SSE 50 ETF constituents"
    )
    parser.add_argument(
        "--mode",
        choices=["incremental", "historical"],
        default="incremental",
        help="incremental mode respects crawl state; historical ignores it",
    )
    parser.add_argument("--start", help="start date YYYY-MM-DD (local Shanghai time)")
    parser.add_argument("--end", help="end date YYYY-MM-DD (inclusive, local Shanghai time)")
    parser.add_argument("--limit-targets", type=int, help="limit number of targets for testing")
    parser.add_argument(
        "--no-export", action="store_true", help="skip exporting consolidated CSV after crawling"
    )
    parser.add_argument(
        "--enable-sina-search",
        action="store_true",
        help="enable Sina search source (disabled by default)",
    )
    parser.add_argument(
        "--sina-pages", type=int, default=3, help="number of Sina search pages per keyword"
    )
    parser.add_argument(
        "--sina-keywords", type=int, default=3, help="number of keywords per target for Sina search"
    )
    parser.add_argument("--disable-feed", action="store_true", help="skip Sina rolling feed source")
    parser.add_argument(
        "--disable-eastmoney", action="store_true", help="skip Eastmoney news source"
    )
    parser.add_argument("--disable-caixin", action="store_true", help="skip Caixin news source")
    parser.add_argument(
        "--sina-feed-pages", type=int, default=3, help="pages per Sina feed channel (default 3)"
    )
    parser.add_argument(
        "--sina-corp-pages", type=int, default=80, help="max pages for Sina corp archive"
    )
    parser.add_argument(
        "--disable-sina-corp", action="store_true", help="skip Sina corp archive source"
    )
    parser.add_argument(
        "--cninfo-window-days", type=int, default=120, help="Cninfo window size in days"
    )
    parser.add_argument(
        "--cninfo-pages", type=int, default=200, help="max pages per Cninfo window (0 = unlimited)"
    )
    parser.add_argument("--disable-cninfo", action="store_true", help="skip Cninfo announcements")
    parser.add_argument("--disable-sse", action="store_true", help="skip SSE announcements")
    parser.add_argument(
        "--sse-page-size", type=int, default=50, help="page size for SSE announcements"
    )
    parser.add_argument("--log-level", default="INFO", help="logging level (default: INFO)")
    return parser.parse_args()


def to_datetime(date_str: str | None, is_end: bool = False) -> datetime | None:
    if not date_str:
        return None
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if is_end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.replace(tzinfo=SHANGHAI_TZ).astimezone(timezone.utc)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

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
        sources.append(
            SinaFeedSource(
                pages_per_lid=args.sina_feed_pages,
            )
        )
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

    crawler = NewsCrawler(news_sources=sources)
    start_dt = to_datetime(args.start)
    end_dt = to_datetime(args.end, is_end=True)
    export_enabled = not args.no_export
    metrics = crawler.run(
        mode=args.mode,
        start=start_dt,
        end=end_dt,
        limit_targets=args.limit_targets,
        export=export_enabled,
    )
    for source_name, summary in metrics.items():
        logging.info(
            "[%s] fetched=%s inserted=%s", source_name, summary["fetched"], summary["inserted"]
        )
    if export_enabled:
        logging.info("Latest dataset exported to %s", config.EXPORT_PATH)
    else:
        logging.info("Export disabled (--no-export)")


if __name__ == "__main__":
    main()
