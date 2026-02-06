#!/usr/bin/env python3
# scripts/12_extract_event_price_path.py

from __future__ import annotations

import argparse
from pathlib import Path

from _eventlib import default_data_dir, ensure_dir, extract_event_price_path_long, save_with_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--windows-name", default="event_windows")
    ap.add_argument("--pre-bdays", type=int, default=5)
    ap.add_argument("--post-bdays", type=int, default=10)
    ap.add_argument("--out-name", default="event_price_path_m5_p10")
    args = ap.parse_args()

    tkr = args.ticker.upper()
    data_dir = args.data_dir if args.data_dir else default_data_dir()

    df = extract_event_price_path_long(
        data_dir=data_dir,
        ticker=tkr,
        windows_csv_name=args.windows_name,
        out_pre_bdays=int(args.pre_bdays),
        out_post_bdays=int(args.post_bdays),
    )

    out_dir = data_dir / tkr / "events"
    ensure_dir(out_dir)
    out_csv = out_dir / f"{args.out_name}.csv"

    save_with_meta(
        df,
        out_csv,
        meta={
            "ticker": tkr,
            "notes": ["Long format: one row per (event, offset). Offsets are trading-day offsets around day0_date."],
        },
    )

    print(f"[OK] {tkr}: wrote {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
