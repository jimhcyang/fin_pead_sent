#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_yf_prices.py

Download daily OHLCV (+ Adj Close) from Yahoo Finance via yfinance and save a
single file:

  data/{TICKER}/prices/yf_ohlcv_daily.csv
    - starts at: --start (default 2019-01-01)
    - contains the full window; no separate "raw" file is written.

Rationale: we now keep the full download in one place and let downstream
scripts (technicals, event windows) apply their own cutoffs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def _flatten_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    t = ticker.upper().strip()

    if isinstance(df.columns, pd.MultiIndex):
        # Drop a constant level (common for single ticker)
        for lvl in range(df.columns.nlevels):
            vals = {str(x).upper().strip() for x in df.columns.get_level_values(lvl)}
            if len(vals) == 1:
                df.columns = df.columns.droplevel(lvl)
                break

        # If still MultiIndex, drop a level that looks like the ticker
        if isinstance(df.columns, pd.MultiIndex):
            for lvl in range(df.columns.nlevels):
                vals = {str(x).upper().strip() for x in df.columns.get_level_values(lvl)}
                if vals.issubset({t, t.replace(".", "-"), ""}):
                    df.columns = df.columns.droplevel(lvl)
                    break

    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            parts = [str(x) for x in c if x is not None and str(x).strip() != ""]
            c = "_".join(parts)
        c = str(c).strip().lower().replace(" ", "_")
        if c in {"adjclose", "adj_close"}:
            c = "adj_close"
        cols.append(c)

    df.columns = cols
    return df


def _inclusive_end_for_daily(end_ymd: str) -> str:
    # yfinance daily often behaves like end-exclusive; make user end inclusive
    return (pd.to_datetime(end_ymd) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2021-01-01", help="Analysis window start (kept in file).")
    ap.add_argument("--end", default="2025-12-31", help="Analysis window end (kept in file).")
    ap.add_argument(
        "--buffer-before-months",
        type=int,
        default=15,
        help="Download this many months before --start (for warm-up + early events). Default: 15.",
    )
    ap.add_argument(
        "--buffer-after-months",
        type=int,
        default=1,
        help="Download this many months after --end (to cover post-event windows). Default: 1.",
    )

    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    ticker = args.ticker.upper().strip()
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)

    dl_start = (start_dt - pd.DateOffset(months=int(args.buffer_before_months))).strftime("%Y-%m-%d")
    dl_end = _inclusive_end_for_daily((end_dt + pd.DateOffset(months=int(args.buffer_after_months))).strftime("%Y-%m-%d"))

    print(f"[INFO] Downloading {ticker} daily OHLCV from Yahoo Finance (yfinance)")
    print(f"[INFO] requested start/end : {args.start} .. {args.end} (inclusive)")
    print(f"[INFO] buffer months       : before={args.buffer_before_months} after={args.buffer_after_months}")
    print(f"[INFO] download window     : {dl_start} .. {(end_dt + pd.DateOffset(months=int(args.buffer_after_months))).strftime('%Y-%m-%d')} (yf_end_used={dl_end})")

    df = yf.download(
        tickers=ticker,
        start=dl_start,
        end=dl_end,
        interval="1d",
        auto_adjust=False,
        actions=True,
        group_by="column",
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned by yfinance for {ticker} in {dl_start}..{args.end}")

    df = _flatten_yf_columns(df, ticker).reset_index()

    if "date" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "date"})
        else:
            raise RuntimeError(f"Could not find a date column. Columns: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    df.insert(0, "ticker", ticker)

    required = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns {missing}. Got columns: {df.columns.tolist()}")

    # Keep core + optional extras
    extras = [c for c in ["dividends", "stock_splits"] if c in df.columns]
    core = df[required + extras].copy()

    out_dir = Path(args.outdir) / ticker / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)

    core["date"] = core["date"].dt.date.astype(str)
    out_path = out_dir / "yf_ohlcv_daily.csv"
    core.to_csv(out_path, index=False)

    # clean up legacy raw file to avoid confusion
    legacy_raw = out_dir / "yf_ohlcv_daily_raw.csv"
    if legacy_raw.exists():
        try:
            legacy_raw.unlink()
            print(f"[INFO] Removed legacy file {legacy_raw}")
        except Exception as e:
            print(f"[WARN] Could not remove legacy raw file {legacy_raw}: {e}")

    print(f"[OK] Wrote {len(core):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
