#!/usr/bin/env python3
# scripts/22_extract_event_price_path.py
#
# Extracts adjusted-close price path around each earnings event:
#   pre_date, react_date, +5/+10/+20 trading-day closes after react_date
#
# No intraday OHLC. No volume.

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    return repo_root() / "data"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_prices(data_dir: Path, ticker: str) -> pd.DataFrame:
    p = data_dir / ticker / "prices" / "yf_ohlcv_daily.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing prices: {p}")
    df = pd.read_csv(p)

    required = {"date", "adj_close"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Prices file missing columns {sorted(missing)}: {p}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_anchors(data_dir: Path, ticker: str, name: str) -> pd.DataFrame:
    p = data_dir / ticker / "events" / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing anchors: {p} (run 21_build_event_anchors.py)")
    return pd.read_csv(p)


def _row_at_date(px: pd.DataFrame, d_str: Optional[str]) -> Optional[pd.Series]:
    if not isinstance(d_str, str) or not d_str:
        return None
    d = pd.to_datetime(d_str, errors="coerce").date()
    s = px.loc[px["date"] == d]
    if s.empty:
        return None
    return s.iloc[0]


def _safe(x: Optional[pd.Series], col: str) -> float:
    if x is None:
        return float("nan")
    return float(pd.to_numeric(x.get(col, np.nan), errors="coerce"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--anchors-name", default="earnings_anchors")
    ap.add_argument("--out-name", default="event_price_path")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    anchors = load_anchors(data_dir, ticker, args.anchors_name)
    px = load_prices(data_dir, ticker)

    rows = []
    for _, a in anchors.iterrows():
        pre = _row_at_date(px, a["pre_date"])
        react = _row_at_date(px, a["react_date"])
        p5 = _row_at_date(px, a.get("d5"))
        p10 = _row_at_date(px, a.get("d10"))
        p20 = _row_at_date(px, a.get("d20"))

        if pre is None or react is None or p5 is None or p10 is None or p20 is None:
            continue

        rows.append(
            {
                "ticker": ticker,
                "earnings_date": a["earnings_date"],
                "announce_timing": a["announce_timing"],
                "pre_date": a["pre_date"],
                "react_date": a["react_date"],
                "d5": a["d5"],
                "d10": a["d10"],
                "d20": a["d20"],

                "pre_adj_close": _safe(pre, "adj_close"),
                "react_adj_close": _safe(react, "adj_close"),
                "adj_close_p5": _safe(p5, "adj_close"),
                "adj_close_p10": _safe(p10, "adj_close"),
                "adj_close_p20": _safe(p20, "adj_close"),
            }
        )

    out = pd.DataFrame(rows)

    out_dir = data_dir / ticker / "events"
    ensure_dir(out_dir)
    out_csv = out_dir / f"{args.out_name}.csv"
    out.to_csv(out_csv, index=False)

    meta = {
        "ticker": ticker,
        "rows": int(out.shape[0]),
        "created_at_local": datetime.now().isoformat(),
        "notes": [
            "Close-only: uses adj_close, no OHLC, no volume.",
            "Anchor dates come from 21_build_event_anchors.py.",
        ],
    }
    (out_dir / f"{args.out_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] {ticker}: wrote {out_csv} ({out.shape[0]} rows)")


if __name__ == "__main__":
    main()