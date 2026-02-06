#!/usr/bin/env python3
# scripts/11_build_event_windows.py

from __future__ import annotations

import argparse
from pathlib import Path

from _eventlib import build_event_windows_df, default_data_dir, ensure_dir, save_with_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--pre-bdays", type=int, default=5)
    ap.add_argument("--post-bdays", type=int, default=10)
    ap.add_argument("--out-name", default="event_windows")
    args = ap.parse_args()

    tkr = args.ticker.upper()
    data_dir = args.data_dir if args.data_dir else default_data_dir()

    df = build_event_windows_df(data_dir=data_dir, ticker=tkr, pre_bdays=int(args.pre_bdays), post_bdays=int(args.post_bdays))

    out_dir = data_dir / tkr / "events"
    ensure_dir(out_dir)
    out_csv = out_dir / f"{args.out_name}.csv"

    save_with_meta(
        df,
        out_csv,
        meta={
            "ticker": tkr,
            "notes": [
                "day0_date is the first close that reflects earnings: BMO=same day close, AMC=next trading day close.",
                f"Path window uses trading-day offsets: [-{args.pre_bdays}, +{args.post_bdays}] relative to day0_date.",
            ],
        },
    )

    print(f"[OK] {tkr}: wrote {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
