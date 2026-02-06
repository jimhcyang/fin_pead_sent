#!/usr/bin/env python3
# scripts/13_compute_event_returns.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _eventlib import (
    compute_event_daily_returns,
    compute_event_window_pct_changes,
    default_data_dir,
    ensure_dir,
    save_with_meta,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--price-path-name", default="event_price_path_m5_p10")
    ap.add_argument("--out-windows", default="event_window_returns")
    ap.add_argument("--out-daily", default="event_daily_returns")
    args = ap.parse_args()

    tkr = args.ticker.upper()
    data_dir = args.data_dir if args.data_dir else default_data_dir()

    p = data_dir / tkr / "events" / f"{args.price_path_name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing price path: {p} (run script 12)")

    path_long = pd.read_csv(p)

    win = compute_event_window_pct_changes(path_long, price_col="adj_close")
    daily = compute_event_daily_returns(path_long, price_col="adj_close")

    out_dir = data_dir / tkr / "events"
    ensure_dir(out_dir)

    save_with_meta(win, out_dir / f"{args.out_windows}.csv", meta={"ticker": tkr, "notes": ["Endpoint pct changes for windows."]})
    save_with_meta(daily, out_dir / f"{args.out_daily}.csv", meta={"ticker": tkr, "notes": ["Daily close-to-close returns per offset (decimal + %)."]})

    print(f"[OK] {tkr}: wrote windows={args.out_windows}.csv and daily={args.out_daily}.csv")


if __name__ == "__main__":
    main()
