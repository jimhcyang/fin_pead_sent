#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import default_data_dir, ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None, help="Single ticker (backward compatible).")
    ap.add_argument("--tickers", nargs="*", default=None, help="Multiple tickers.")
    ap.add_argument("--data-dir", default=None, help="Default: repo_root/data")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    tickers: list[str] = []
    if args.ticker:
        tickers.append(args.ticker)
    if args.tickers:
        tickers += args.tickers

    if not tickers:
        raise SystemExit("Provide --ticker or --tickers.")

    for t in tickers:
        ticker = t.upper().strip()
        base = data_dir / ticker
        for sub in ["prices", "technicals", "calendar", "events", "financials", "transcripts", "news"]:
            ensure_dir(base / sub)

        print(f"[OK] Initialized {base}")


if __name__ == "__main__":
    main()
