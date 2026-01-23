#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import default_data_dir, ensure_dir, now_iso, read_csv, write_csv, write_json
from _indicators import sma, ema, rsi, macd, atr, bollinger, obv, cmf, mfi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    prices_dir = data_dir / ticker / "prices"
    prices_trim_path = prices_dir / "yf_ohlcv_daily.csv"
    prices_raw_path = prices_dir / "yf_ohlcv_daily_raw.csv"

    if not prices_trim_path.exists():
        raise RuntimeError(f"Missing prices file: {prices_trim_path}. Run 01_yf_prices.py first.")

    # Trimmed window defines what we save (alignment for joins)
    df_trim = read_csv(prices_trim_path)
    df_trim["date"] = pd.to_datetime(df_trim["date"])
    df_trim = df_trim.sort_values("date").reset_index(drop=True)

    analysis_start = df_trim["date"].min()
    analysis_end = df_trim["date"].max()

    # Raw (buffered) is used to compute indicators (warm-up)
    if prices_raw_path.exists():
        df = read_csv(prices_raw_path)
        source_prices = "Yahoo Finance (via yfinance) [raw with buffer]"
    else:
        df = df_trim.copy()
        source_prices = "Yahoo Finance (via yfinance) [trimmed only]"

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Use adjusted close for returns
    adj = df["adj_close"].astype(float)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    vol = df["volume"].astype(float)

    out = pd.DataFrame({"date": df["date"]})

    # Returns & drift helpers
    out["ret_1d"] = adj.pct_change()
    out["logret_1d"] = np.log(adj / adj.shift(1))
    out["ret_5d"] = adj.pct_change(5)
    out["ret_20d"] = adj.pct_change(20)

    # Trend
    for w in [5, 20, 50]:
        out[f"sma_{w}"] = sma(adj, w)
        out[f"ema_{w}"] = ema(adj, w)

    # MACD
    macd_line, sig_line, hist = macd(adj, 12, 26, 9)
    out["macd_12_26"] = macd_line
    out["macd_signal_9"] = sig_line
    out["macd_hist"] = hist

    # Momentum
    out["rsi_14"] = rsi(adj, 14)
    out["roc_10"] = adj.pct_change(10)
    out["mom_10"] = adj - adj.shift(10)

    # Volatility
    out["vol_20"] = out["logret_1d"].rolling(20, min_periods=20).std()
    out["vol_60"] = out["logret_1d"].rolling(50, min_periods=50).std()
    out["atr_14"] = atr(high, low, close, 14)

    # Bands
    mid, upper, lower, width = bollinger(adj, 20, 2.0)
    out["bb_mid_20"] = mid
    out["bb_upper_20_2"] = upper
    out["bb_lower_20_2"] = lower
    out["bb_width_20_2"] = width

    # Volume/flow
    out["obv"] = obv(adj, vol)
    out["cmf_20"] = cmf(high, low, close, vol, 20)
    out["mfi_14"] = mfi(high, low, close, vol, 14)

    # Range proxies
    out["hl_range"] = (high - low) / close.replace(0, np.nan)
    out["oc_gap"] = (open_ - close) / close.replace(0, np.nan)

    # Truncate output back to analysis window (so it aligns with event panel)
    out = out[(out["date"] >= analysis_start) & (out["date"] <= analysis_end)].copy()
    out["date"] = out["date"].dt.date.astype(str)

    out_dir = data_dir / ticker / "technicals"
    ensure_dir(out_dir)

    out_csv = out_dir / "technicals_daily.csv"
    write_csv(out, out_csv)

    meta = {
        "ticker": ticker,
        "source_prices": source_prices,
        "generated_at_et": now_iso(),
        "rows": int(out.shape[0]),
        "analysis_window_start": analysis_start.date().isoformat(),
        "analysis_window_end": analysis_end.date().isoformat(),
        "notes": "Indicators computed from buffered daily OHLCV when available; saved output is truncated to the analysis window defined by yf_ohlcv_daily.csv. Returns use adjusted close.",
    }
    write_json(meta, out_dir / "technicals_daily.meta.json")

    print(f"[OK] Wrote {out_csv} ({out.shape[0]} rows)")


if __name__ == "__main__":
    main()
