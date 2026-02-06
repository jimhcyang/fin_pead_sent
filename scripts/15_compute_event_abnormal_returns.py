#!/usr/bin/env python3
# scripts/15_compute_event_abnormal_returns.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _eventlib import (
    WINDOWS,
    compute_window_car,
    default_data_dir,
    ensure_dir,
    load_market,
    load_prices,
    save_with_meta,
    window_return_offsets,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--daily-name", default="event_daily_returns")
    ap.add_argument("--mm-name", default="event_market_model")
    ap.add_argument("--market-rel", default="_tmp_market/spx/prices/yf_ohlcv_daily.csv")
    ap.add_argument("--out-name", default="event_abnormal_windows")
    ap.add_argument("--out-daily", default="event_abnormal_daily")
    args = ap.parse_args()

    tkr = args.ticker.upper()
    data_dir = args.data_dir if args.data_dir else default_data_dir()

    daily_path = data_dir / tkr / "events" / f"{args.daily_name}.csv"
    mm_path = data_dir / tkr / "events" / f"{args.mm_name}.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing daily returns: {daily_path} (run script 13)")
    if not mm_path.exists():
        raise FileNotFoundError(f"Missing market model: {mm_path} (run script 14)")

    daily = pd.read_csv(daily_path)
    mm = pd.read_csv(mm_path)

    px = load_prices(data_dir, tkr)
    mkt = load_market(data_dir, rel=args.market_rel)
    mkt_ret = mkt["adj_close"].pct_change()

    # Join alpha/beta into daily rows (by earnings_date)
    mm_small = mm[["earnings_date", "alpha_1y", "beta_1y", "alpha_1q", "beta_1q"]].copy()
    daily = daily.merge(mm_small, on="earnings_date", how="left")

    # compute market return for each daily row using its date (return is for that date move)
    daily["date_dt"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["mkt_ret"] = daily["date_dt"].map(lambda d: float(mkt_ret.get(d, np.nan)) if pd.notna(d) else np.nan)

    # daily abnormal returns (decimal)
    daily["abn_simple"] = daily["ret"] - daily["mkt_ret"]
    daily["abn_mm_1y"] = daily["ret"] - (daily["alpha_1y"] + daily["beta_1y"] * daily["mkt_ret"])
    daily["abn_mm_1q"] = daily["ret"] - (daily["alpha_1q"] + daily["beta_1q"] * daily["mkt_ret"])

    # per-event CARs over windows
    out_rows = []
    for (tkr2, ed), g in daily.groupby(["ticker", "earnings_date"], sort=True):
        # map abnormal returns by offset (return at offset k is move from k-1 -> k)
        abn_simple = dict(zip(g["offset"].astype(int).tolist(), g["abn_simple"].tolist()))
        abn_1y = dict(zip(g["offset"].astype(int).tolist(), g["abn_mm_1y"].tolist()))
        abn_1q = dict(zip(g["offset"].astype(int).tolist(), g["abn_mm_1q"].tolist()))

        rec = {"ticker": tkr2, "earnings_date": ed}

        for name, (a, b) in WINDOWS.items():
            # CAR in percent points = 100 * sum(daily abnormal returns in decimal)
            car_s = compute_window_car(abn_simple, a, b)
            car_y = compute_window_car(abn_1y, a, b)
            car_q = compute_window_car(abn_1q, a, b)
            rec[f"car_simple_{name}_pct"] = 100.0 * car_s if np.isfinite(car_s) else np.nan
            rec[f"car_mm_1y_{name}_pct"] = 100.0 * car_y if np.isfinite(car_y) else np.nan
            rec[f"car_mm_1q_{name}_pct"] = 100.0 * car_q if np.isfinite(car_q) else np.nan

        out_rows.append(rec)

    out = pd.DataFrame(out_rows).sort_values(["ticker", "earnings_date"]).reset_index(drop=True)

    out_dir = data_dir / tkr / "events"
    ensure_dir(out_dir)

    save_with_meta(out, out_dir / f"{args.out_name}.csv", meta={"ticker": tkr, "notes": ["Window CARs for simple and market-model abnormal returns."]})
    save_with_meta(
        daily.drop(columns=["date_dt"], errors="ignore"),
        out_dir / f"{args.out_daily}.csv",
        meta={"ticker": tkr, "notes": ["Daily abnormal returns (decimal) at each offset."]},
    )

    print(f"[OK] {tkr}: wrote {args.out_name}.csv and {args.out_daily}.csv")


if __name__ == "__main__":
    main()
