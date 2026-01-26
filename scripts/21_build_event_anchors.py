#!/usr/bin/env python3
# scripts/21_build_event_anchors.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    return repo_root() / "data"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_calendar(data_dir: Path, ticker: str) -> pd.DataFrame:
    p = data_dir / ticker / "calendar" / "earnings_calendar.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing calendar: {p}")
    df = pd.read_csv(p)
    if "earnings_date" not in df.columns:
        raise RuntimeError(f"Calendar missing 'earnings_date': {p}")
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    df = df.dropna(subset=["earnings_date"]).sort_values("earnings_date").reset_index(drop=True)
    return df


def load_trading_dates(data_dir: Path, ticker: str) -> list:
    p = data_dir / ticker / "prices" / "yf_ohlcv_daily.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing prices: {p} (run 01_yf_prices.py first)")
    df = pd.read_csv(p)
    if "date" not in df.columns:
        raise RuntimeError(f"Prices missing 'date': {p}")
    d = pd.to_datetime(df["date"], errors="coerce").dt.date
    d = d.dropna().sort_values().unique().tolist()
    return d


def asof_trading_date(trading: list, d) -> int:
    """
    Return index i such that trading[i] is the last trading date <= d.
    Raises if d is earlier than first trading date.
    """
    # binary search
    lo, hi = 0, len(trading) - 1
    if d < trading[0]:
        raise ValueError(f"date {d} is before first trading date {trading[0]}")
    while lo <= hi:
        mid = (lo + hi) // 2
        if trading[mid] <= d:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi


def anchor_dates(trading: list, earnings_date, timing: str) -> Tuple:
    """
    Given earnings_date and announce timing, return:
      pre_date, react_date, d5, d10, d20 as strings YYYY-MM-DD.
    Conventions:
      - BMO: pre_date = prev trading day close; react_date = same-day close
      - AMC: pre_date = same-day close; react_date = next trading day close
    Then d5/d10/d20 are trading-day offsets from react_date.
    """
    i = asof_trading_date(trading, earnings_date)

    if timing.lower() == "bmo":
        pre_i = max(i - 1, 0)
        react_i = i
    else:
        # default to AMC
        pre_i = i
        react_i = min(i + 1, len(trading) - 1)

    def idx_to_str(j: int) -> str:
        return pd.to_datetime(trading[j]).strftime("%Y-%m-%d")

    pre = idx_to_str(pre_i)
    react = idx_to_str(react_i)

    def offset(j: int, k: int) -> str:
        jj = min(j + k, len(trading) - 1)
        return idx_to_str(jj)

    d5 = offset(react_i, 5)
    d10 = offset(react_i, 10)
    d20 = offset(react_i, 20)

    return pre, react, d5, d10, d20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-name", default="earnings_anchors")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    cal = load_calendar(data_dir, ticker)
    trading = load_trading_dates(data_dir, ticker)

    # determine timing column
    timing_col = None
    for c in ["announce_timing", "time", "timing", "when"]:
        if c in cal.columns:
            timing_col = c
            break
    if timing_col is None:
        # default to AMC if absent
        cal["announce_timing"] = "amc"
        timing_col = "announce_timing"

    rows = []
    for _, r in cal.iterrows():
        ed = r["earnings_date"]
        timing = str(r.get(timing_col, "amc") or "amc").strip().lower()
        if timing not in ("bmo", "amc"):
            timing = "amc"

        pre, react, d5, d10, d20 = anchor_dates(trading, ed, timing)

        rows.append(
            {
                "ticker": ticker,
                "earnings_date": pd.to_datetime(ed).strftime("%Y-%m-%d"),
                "announce_timing": timing,
                "pre_date": pre,
                "react_date": react,
                "d5": d5,
                "d10": d10,
                "d20": d20,
            }
        )

    df = pd.DataFrame(rows)

    out_dir = data_dir / ticker / "events"
    ensure_dir(out_dir)
    out_csv = out_dir / f"{args.out_name}.csv"
    df.to_csv(out_csv, index=False)

    meta = {
        "ticker": ticker,
        "rows": int(df.shape[0]),
        "created_at_local": datetime.now().isoformat(),
        "notes": [
            "Anchors built from ticker trading dates in prices/yf_ohlcv_daily.csv",
            "BMO: pre=prev trading close, react=same-day close",
            "AMC: pre=same-day close, react=next trading close",
            "d5/d10/d20 are trading-day offsets from react_date",
        ],
    }
    (out_dir / f"{args.out_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] {ticker}: wrote {out_csv} ({df.shape[0]} rows)")


if __name__ == "__main__":
    main()
