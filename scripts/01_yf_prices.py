#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_yf_prices.py

Download daily OHLCV (+ Adj Close) from Yahoo Finance via yfinance and save ONE file:

  data/{TICKER}/prices/yf_ohlcv_daily.csv

This script is robust to yfinance returning MultiIndex (tuple) columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def _flatten_yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance may return MultiIndex columns (tuples). Flatten to single strings.
    For single-ticker downloads, we usually can drop the ticker level.
    """
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

    # Flatten any remaining tuples and standardize names
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
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--outdir", default="data")  # repo_root/data by default if you run from repo root
    args = ap.parse_args()

    ticker = args.ticker.upper().strip()
    start = args.start
    end = args.end
    end_for_yf = _inclusive_end_for_daily(end)

    print(f"[INFO] Downloading {ticker} daily OHLCV from Yahoo Finance (yfinance)")
    print(f"[INFO] start={start} end={end} (yf_end_used={end_for_yf})")

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end_for_yf,
        interval="1d",
        auto_adjust=False,
        actions=True,       # dividends/splits if available
        group_by="column",
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned by yfinance for {ticker} in {start}..{end}")

    df = _flatten_yf_columns(df, ticker)
    df = df.reset_index()

    # Standardize index column name to 'date'
    if "date" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "date"})
        else:
            raise RuntimeError(f"Could not find a date column. Columns: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    df.insert(0, "ticker", ticker)

    # Keep a clean canonical set of columns (extra columns like dividends/splits may exist)
    required = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns {missing}. Got columns: {df.columns.tolist()}")

    # Preserve dividends/splits if present, but always include the required core fields
    core = df[required].copy()
    extras = []
    for c in ["dividends", "stock_splits"]:
        if c in df.columns:
            extras.append(c)
    if extras:
        core = pd.concat([core, df[extras]], axis=1)

    out_dir = Path(args.outdir) / ticker / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "yf_ohlcv_daily.csv"
    core.to_csv(out_path, index=False)

    print(f"[SUCCESS] Wrote {len(core):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
