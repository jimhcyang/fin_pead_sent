#!/usr/bin/env python3
# scripts/25_yf_download_spx.py
#
# Close-only market proxy download for abnormal returns:
#   date, adj_close
#
# Keeps filename "yf_ohlcv_daily.csv" for compatibility.

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    return repo_root() / "data"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns.to_flat_index()]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--symbol", default="^GSPC")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default="2026-01-31")
    ap.add_argument("--out-rel", default="_tmp_market/spx/prices/yf_ohlcv_daily.csv")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    out_path = data_dir / args.out_rel
    ensure_dir(out_path.parent)

    if out_path.exists() and not args.force:
        print(f"[SKIP] SPX exists: {out_path} (use --force to redownload)")
        return

    import yfinance as yf

    df = yf.download(
        args.symbol,
        start=args.start,
        end=args.end,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {args.symbol} in [{args.start}, {args.end}]")

    df = _flatten_yf_columns(df).reset_index()
    if "Date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "Date"})

    df = df.rename(columns={"Date": "date", "Adj Close": "adj_close", "Close": "close"})
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["adj_close"] = pd.to_numeric(df[["adj_close"]].squeeze(), errors="coerce")

    df = df[["date", "adj_close"]].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df.to_csv(out_path, index=False)

    meta = {
        "symbol": args.symbol,
        "rows": int(df.shape[0]),
        "start": args.start,
        "end": args.end,
        "created_at_local": datetime.now().isoformat(),
        "schema": ["date", "adj_close"],
    }
    (out_path.parent / "yf_ohlcv_daily.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] downloaded {args.symbol} -> {out_path} ({df.shape[0]} rows)")


if __name__ == "__main__":
    main()
