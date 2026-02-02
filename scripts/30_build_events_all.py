#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
30_build_events_all.py (v3)

1) Ensure market prices exist (25_yf_download_spx.py)  **PASS --data-dir**
2) For each ticker:
   - 29_build_event_study_panel.py
   - 24_merge_event_fundamentals.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd):
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--tickers", nargs="*", required=True)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--skip-market", action="store_true")
    ap.add_argument("--keep-last-event", action="store_true")
    ap.add_argument("--stable-period", choices=["quarter", "annual"], default="quarter")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent

    market_rel = Path("_tmp_market/spx/prices/yf_ohlcv_daily.csv")
    market_abs = args.data_dir / market_rel

    if not args.skip_market:
        run([
            "python", str(script_dir / "25_yf_download_spx.py"),
            "--data-dir", str(args.data_dir),          # <-- KEY FIX
            "--symbol", "^GSPC",
            "--start", args.start,
            "--end", args.end,
            "--out-rel", str(market_rel),
        ])

    if not market_abs.exists():
        raise FileNotFoundError(
            f"SPX market file not found at: {market_abs}\n"
            f"Expected because 25_yf_download_spx writes into data_dir/out_rel.\n"
            f"data_dir={args.data_dir} out_rel={market_rel}"
        )

    for t in args.tickers:
        tkr = t.upper()

        run([
            "python", str(script_dir / "29_build_event_study_panel.py"),
            "--ticker", tkr,
            "--data-dir", str(args.data_dir),
            "--market-prices", str(market_abs),
            *(["--keep-last-event"] if args.keep_last_event else []),
        ])

        run([
            "python", str(script_dir / "24_merge_event_fundamentals.py"),
            "--ticker", tkr,
            "--data-dir", str(args.data_dir),
            "--period", args.stable_period,
        ])

    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
