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

    prices_path = data_dir / ticker / "prices" / "yf_ohlcv_daily.csv"
    if not prices_path.exists():
        raise RuntimeError(f"Missing prices file: {prices_path}. Run 01_yf_prices.py first.")

    df = read_csv(prices_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Use adjusted close for returns
    adj = df["adj_close"].astype(float)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    vol = df["volume"].astype(float)

    out = pd.DataFrame({"date": df["date"].dt.date.astype(str)})

    # Returns & drift helpers
    out["ret_1d"] = adj.pct_change()
    out["logret_1d"] = np.log(adj / adj.shift(1))
    out["ret_5d"] = adj.pct_change(5)
    out["ret_20d"] = adj.pct_change(20)
    out["ret_60d"] = adj.pct_change(60)

    # Drawdown
    roll_max = adj.rolling(252, min_periods=252).max()
    out["drawdown_252"] = adj / roll_max - 1

    # Trend
    for w in [5, 10, 20, 50, 100, 200]:
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
    out["vol_60"] = out["logret_1d"].rolling(60, min_periods=60).std()
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

    out_dir = data_dir / ticker / "technicals"
    ensure_dir(out_dir)

    out_csv = out_dir / "technicals_daily.csv"
    write_csv(out, out_csv)

    meta = {
        "ticker": ticker,
        "source_prices": "Yahoo Finance (via yfinance)",
        "generated_at_et": now_iso(),
        "rows": int(out.shape[0]),
        "notes": "All indicators computed from daily OHLCV; returns use adjusted close.",
    }
    write_json(meta, out_dir / "technicals_daily.meta.json")

    print(f"[OK] Wrote {out_csv} ({out.shape[0]} rows)")


if __name__ == "__main__":
    main()
