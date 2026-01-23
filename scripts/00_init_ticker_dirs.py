#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import default_data_dir, ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None, help="Default: repo_root/data")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    base = data_dir / ticker
    for sub in ["prices", "technicals", "calendar", "events", "transcripts", "news"]:
        ensure_dir(base / sub)

    print(f"[OK] Initialized {base}")


if __name__ == "__main__":
    main()
