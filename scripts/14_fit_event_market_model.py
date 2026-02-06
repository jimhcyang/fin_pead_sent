#!/usr/bin/env python3
# scripts/14_fit_event_market_model.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _eventlib import default_data_dir, ensure_dir, fit_alpha_beta, load_market, load_prices, save_with_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--windows-name", default="event_windows")
    ap.add_argument("--market-rel", default="_tmp_market/spx/prices/yf_ohlcv_daily.csv")
    ap.add_argument("--year-len", type=int, default=252)
    ap.add_argument("--qtr-len", type=int, default=63)
    ap.add_argument("--out-name", default="event_market_model")
    args = ap.parse_args()

    tkr = args.ticker.upper()
    data_dir = args.data_dir if args.data_dir else default_data_dir()

    wpath = data_dir / tkr / "events" / f"{args.windows_name}.csv"
    if not wpath.exists():
        raise FileNotFoundError(f"Missing windows file: {wpath} (run script 11)")
    w = pd.read_csv(wpath)

    px = load_prices(data_dir, tkr)
    mkt = load_market(data_dir, rel=args.market_rel)

    stock_ret = px["adj_close"].pct_change()
    mkt_ret = mkt["adj_close"].pct_change()

    rows = []
    for _, ev in w.iterrows():
        day0 = pd.to_datetime(ev["day0_date"], errors="coerce")
        if pd.isna(day0) or day0 not in px.index:
            continue

        i0 = int(px.index.get_indexer([day0])[0])
        i_end = i0 - 1  # estimation ends at day(-1)
        if i_end < 0:
            continue

        def slice_rets(L: int):
            i_start = i_end - (L - 1)
            if i_start < 1:
                return None, None, 0
            idx = px.index[i_start : i_end + 1]
            ri = stock_ret.loc[idx].to_numpy(dtype=float)
            rm = mkt_ret.reindex(idx).to_numpy(dtype=float)
            return ri, rm, int(len(idx))

        ri_y, rm_y, n_y = slice_rets(int(args.year_len))
        ri_q, rm_q, n_q = slice_rets(int(args.qtr_len))

        a_y, b_y, r2_y, s_y, _ = fit_alpha_beta(ri_y, rm_y) if n_y > 0 else (np.nan, np.nan, np.nan, np.nan, 0)
        a_q, b_q, r2_q, s_q, _ = fit_alpha_beta(ri_q, rm_q) if n_q > 0 else (np.nan, np.nan, np.nan, np.nan, 0)

        rows.append(
            {
                "ticker": tkr,
                "earnings_date": ev["earnings_date"],
                "day0_date": ev["day0_date"],
                "est_end_date": px.index[i_end].strftime("%Y-%m-%d"),

                "alpha_1y": a_y,
                "beta_1y": b_y,
                "r2_1y": r2_y,
                "resid_vol_1y": s_y,
                "n_1y": n_y,

                "alpha_1q": a_q,
                "beta_1q": b_q,
                "r2_1q": r2_q,
                "resid_vol_1q": s_q,
                "n_1q": n_q,
            }
        )

    out = pd.DataFrame(rows).sort_values(["ticker", "earnings_date"]).reset_index(drop=True)

    out_dir = data_dir / tkr / "events"
    ensure_dir(out_dir)
    out_csv = out_dir / f"{args.out_name}.csv"

    save_with_meta(
        out,
        out_csv,
        meta={
            "ticker": tkr,
            "market_rel": args.market_rel,
            "notes": [
                "Market model estimated per event on daily returns up to day(-1) relative to day0_date.",
                f"1y uses {args.year_len} trading days; 1q uses {args.qtr_len} trading days.",
            ],
        },
    )

    print(f"[OK] {tkr}: wrote {out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
