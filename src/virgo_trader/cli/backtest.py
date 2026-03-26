"""Backtest a trained model (CLI).

This command is intended to replace legacy ad-hoc scripts with a packaged,
installable entrypoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a trained Virgo model.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (without .zip) or a path to a .zip checkpoint.",
    )
    parser.add_argument("--start_date", required=True, help="Backtest start date (YYYYMMDD).")
    parser.add_argument("--end_date", required=True, help="Backtest end date (YYYYMMDD).")
    parser.add_argument(
        "--output_label",
        default="backtest",
        help="Label used in the report filename (default: backtest).",
    )
    parser.add_argument(
        "--sentiment_dataset",
        default="",
        help="Optional sentiment JSONL path (overrides model training metadata).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write the report under the reports directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the JSON payload (defaults to stdout).",
    )
    return parser.parse_args(argv)


def _resolve_model_arg(model_arg: str) -> tuple[str, Optional[Path]]:
    raw = str(model_arg).strip()
    if not raw:
        raise ValueError("--model is required.")

    # Heuristic: treat paths or *.zip values as a path.
    path_like = raw.endswith(".zip") or ("/" in raw) or ("\\" in raw)
    if path_like:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        return path.stem, path

    name = raw[:-4] if raw.lower().endswith(".zip") else raw
    return name, None


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Delay heavy imports so `--help` works in minimal environments.
    from virgo_trader.backtest.runner import run_backtest

    model_name, model_path = _resolve_model_arg(args.model)
    payload = run_backtest(
        model_name=model_name,
        model_path=model_path,
        start_date=args.start_date,
        end_date=args.end_date,
        output_label=args.output_label,
        sentiment_dataset=args.sentiment_dataset.strip() or None,
        seed=args.seed,
        write_report=not bool(args.no_report),
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(args.out))
        return 0

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
