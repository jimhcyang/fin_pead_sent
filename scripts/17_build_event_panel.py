#!/usr/bin/env python3
# scripts/17_build_event_panel.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _eventlib import default_data_dir, ensure_dir, save_with_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--windows-returns", default="event_window_returns")
    ap.add_argument("--abnormal-windows", default="event_abnormal_windows")
    ap.add_argument("--features", default="event_features_stable")
    ap.add_argument("--out-name", default="event_panel")
    args = ap.parse_args()

    tkr = args.ticker.upper()
    data_dir = args.data_dir if args.data_dir else default_data_dir()
    edir = data_dir / tkr / "events"

    r = pd.read_csv(edir / f"{args.windows_returns}.csv")
    a = pd.read_csv(edir / f"{args.abnormal_windows}.csv")
    f = pd.read_csv(edir / f"{args.features}.csv")

    # merge by earnings_date
    out = r.merge(a, on=["ticker", "earnings_date"], how="left")
    out = out.merge(f.drop(columns=["ticker"], errors="ignore"), on=["earnings_date"], how="left")

    out_dir = edir
    ensure_dir(out_dir)
    out_csv = out_dir / f"{args.out_name}.csv"

    save_with_meta(out, out_csv, meta={"ticker": tkr, "notes": ["One row per earnings event: window returns + CARs + stable features."]})
    print(f"[OK] {tkr}: wrote {out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
