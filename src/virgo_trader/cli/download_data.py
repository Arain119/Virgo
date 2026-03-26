"""Pre-download and optionally feature-engineer datasets (CLI)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-download stock data.")
    parser.add_argument(
        "--stock_pool", type=str, default="sse50", help="Stock pool to download (e.g., sse50)."
    )
    parser.add_argument("--start_date", type=str, default="20220101", help="Start date.")
    parser.add_argument("--end_date", type=str, default="20260101", help="End date.")
    parser.add_argument(
        "--use_live_pool",
        action="store_true",
        help="Use live AkShare pool constituents (default: False; uses fallback list).",
    )
    parser.add_argument(
        "--prefer_qfq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer qfq-adjusted data sources.",
    )
    parser.add_argument(
        "--build_features",
        action="store_true",
        help="Also build and cache feature-engineered datasets for both calendar modes.",
    )
    args = parser.parse_args(argv)

    # Delay project imports so `--help` works in minimal environments.
    from virgo_trader.data.data_fetcher import get_stock_data
    from virgo_trader.data.feature_cache import feature_cache_path, save_cached_features
    from virgo_trader.data.feature_engineer import FeatureEngineer
    from virgo_trader.data.stock_pools import get_stock_pool_codes

    print(f"Fetching data for {args.stock_pool} from {args.start_date} to {args.end_date}...")

    try:
        codes = list(get_stock_pool_codes(args.stock_pool, use_live=args.use_live_pool))
    except Exception as exc:
        print(f"Error loading stock pool '{args.stock_pool}': {exc}")
        codes = ["510050.SH"]

    print(f"Found {len(codes)} symbols.")

    engineer_cal0 = FeatureEngineer(use_calendar_features=False) if args.build_features else None
    engineer_cal1 = FeatureEngineer(use_calendar_features=True) if args.build_features else None

    for i, code in enumerate(codes):
        print(f"[{i + 1}/{len(codes)}] Check/Download {code}...")
        try:
            df = get_stock_data(
                code, args.start_date, args.end_date, prefer_qfq=bool(args.prefer_qfq)
            )
            if df is None or df.empty:
                print(f"  Warning: Empty data for {code}")
                continue

            if args.build_features and engineer_cal0 and engineer_cal1:
                for use_calendar_features, engineer in (
                    (False, engineer_cal0),
                    (True, engineer_cal1),
                ):
                    cache_path = feature_cache_path(
                        symbol=code,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        use_calendar_features=use_calendar_features,
                    )
                    if cache_path.exists():
                        continue
                    processed = engineer.process(df.copy())
                    save_cached_features(cache_path, processed)
        except Exception as exc:
            print(f"  Error fetching {code}: {exc}")

    print("Data download completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
