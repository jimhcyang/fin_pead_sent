#!/usr/bin/env python3
# scripts/24_merge_event_fundamentals.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    return repo_root() / "data"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# (Same scored feature lists you already liked — unchanged)
# -------------------------------------------------------------------

KM_FEATURES_SCORED = [
    ("freeCashFlowYield", 93, "PEAD 36/40, XSec 24/25, Quality 14/15, Interp 9/10, NonRed 10/10"),
    ("earningsYield", 92,     "PEAD 36/40, XSec 24/25, Quality 14/15, Interp 8/10, NonRed 10/10"),
    ("enterpriseValueOverEBITDA", 90, "PEAD 33/40, XSec 24/25, Quality 14/15, Interp 9/10, NonRed 10/10"),
    ("peRatio", 88,           "PEAD 32/40, XSec 23/25, Quality 14/15, Interp 9/10, NonRed 10/10"),
    ("pfcfRatio", 87,         "PEAD 33/40, XSec 23/25, Quality 13/15, Interp 8/10, NonRed 10/10"),
    ("pocfratio", 86,         "PEAD 32/40, XSec 22/25, Quality 13/15, Interp 9/10, NonRed 10/10"),
    ("priceToSalesRatio", 84, "PEAD 30/40, XSec 22/25, Quality 14/15, Interp 9/10, NonRed 9/10"),
    ("evToSales", 83,         "PEAD 29/40, XSec 22/25, Quality 14/15, Interp 9/10, NonRed 9/10"),
    ("pbRatio", 82,           "PEAD 28/40, XSec 22/25, Quality 14/15, Interp 9/10, NonRed 9/10"),
    ("roic", 82,              "PEAD 33/40, XSec 20/25, Quality 13/15, Interp 8/10, NonRed 8/10"),
    ("incomeQuality", 81,     "PEAD 32/40, XSec 19/25, Quality 13/15, Interp 8/10, NonRed 9/10"),
    ("netDebtToEBITDA", 79,   "PEAD 29/40, XSec 20/25, Quality 13/15, Interp 8/10, NonRed 9/10"),
    ("debtToEquity", 78,      "PEAD 28/40, XSec 20/25, Quality 13/15, Interp 8/10, NonRed 9/10"),
    ("capexToRevenue", 76,    "PEAD 28/40, XSec 18/25, Quality 13/15, Interp 8/10, NonRed 9/10"),
    ("researchAndDdevelopementToRevenue", 74, "PEAD 27/40, XSec 18/25, Quality 12/15, Interp 8/10, NonRed 9/10"),
    ("freeCashFlowPerShare", 73,     "PEAD 27/40, XSec 17/25, Quality 13/15, Interp 8/10, NonRed 8/10"),
    ("operatingCashFlowPerShare", 72,"PEAD 26/40, XSec 17/25, Quality 13/15, Interp 8/10, NonRed 8/10"),
    ("revenuePerShare", 70,          "PEAD 24/40, XSec 16/25, Quality 13/15, Interp 9/10, NonRed 8/10"),
    ("cashPerShare", 68,             "PEAD 23/40, XSec 16/25, Quality 13/15, Interp 8/10, NonRed 8/10"),
    ("bookValuePerShare", 67,        "PEAD 22/40, XSec 16/25, Quality 13/15, Interp 8/10, NonRed 8/10"),
    ("enterpriseValue", 64,          "PEAD 20/40, XSec 16/25, Quality 15/15, Interp 7/10, NonRed 6/10"),
    ("marketCap", 63,                "PEAD 19/40, XSec 16/25, Quality 15/15, Interp 7/10, NonRed 6/10"),
]

RT_FEATURES_SCORED = [
    ("operatingProfitMargin", 90, "PEAD 34/40, XSec 22/25, Quality 14/15, Interp 10/10, NonRed 10/10"),
    ("netProfitMargin", 89,       "PEAD 34/40, XSec 22/25, Quality 14/15, Interp 10/10, NonRed 9/10"),
    ("grossProfitMargin", 86,     "PEAD 32/40, XSec 21/25, Quality 14/15, Interp 10/10, NonRed 9/10"),
    ("returnOnEquity", 86,        "PEAD 33/40, XSec 21/25, Quality 13/15, Interp 9/10, NonRed 10/10"),
    ("returnOnAssets", 84,        "PEAD 32/40, XSec 20/25, Quality 13/15, Interp 9/10, NonRed 10/10"),
    ("returnOnCapitalEmployed", 83,"PEAD 32/40, XSec 20/25, Quality 13/15, Interp 8/10, NonRed 10/10"),
    ("ebitPerRevenue", 82,        "PEAD 31/40, XSec 20/25, Quality 13/15, Interp 9/10, NonRed 9/10"),
    ("interestCoverage", 81,      "PEAD 30/40, XSec 19/25, Quality 13/15, Interp 9/10, NonRed 10/10"),
    ("debtEquityRatio", 80,       "PEAD 29/40, XSec 20/25, Quality 13/15, Interp 9/10, NonRed 9/10"),
    ("debtRatio", 78,             "PEAD 28/40, XSec 19/25, Quality 13/15, Interp 9/10, NonRed 9/10"),
    ("cashFlowToDebtRatio", 78,   "PEAD 29/40, XSec 18/25, Quality 13/15, Interp 9/10, NonRed 9/10"),
    ("operatingCashFlowSalesRatio", 77, "PEAD 28/40, XSec 18/25, Quality 13/15, Interp 9/10, NonRed 9/10"),
    ("enterpriseValueMultiple", 76,"PEAD 28/40, XSec 20/25, Quality 14/15, Interp 8/10, NonRed 6/10"),
    ("effectiveTaxRate", 74,      "PEAD 24/40, XSec 17/25, Quality 14/15, Interp 10/10, NonRed 9/10"),
    ("priceEarningsRatio", 74,    "PEAD 26/40, XSec 19/25, Quality 14/15, Interp 8/10, NonRed 7/10"),
    ("currentRatio", 73,          "PEAD 24/40, XSec 17/25, Quality 14/15, Interp 9/10, NonRed 9/10"),
    ("priceToFreeCashFlowsRatio", 72,"PEAD 26/40, XSec 18/25, Quality 13/15, Interp 8/10, NonRed 7/10"),
    ("quickRatio", 72,            "PEAD 24/40, XSec 16/25, Quality 14/15, Interp 9/10, NonRed 9/10"),
    ("cashRatio", 70,             "PEAD 23/40, XSec 16/25, Quality 14/15, Interp 9/10, NonRed 8/10"),
    ("priceToOperatingCashFlowsRatio", 70,"PEAD 25/40, XSec 17/25, Quality 13/15, Interp 8/10, NonRed 7/10"),
    ("cashConversionCycle", 69,   "PEAD 24/40, XSec 15/25, Quality 14/15, Interp 8/10, NonRed 8/10"),
    ("daysOfSalesOutstanding", 68,"PEAD 23/40, XSec 15/25, Quality 14/15, Interp 8/10, NonRed 8/10"),
    ("daysOfInventoryOutstanding", 67,"PEAD 23/40, XSec 15/25, Quality 14/15, Interp 7/10, NonRed 8/10"),
    ("priceFairValue", 66,        "PEAD 22/40, XSec 14/25, Quality 13/15, Interp 9/10, NonRed 8/10"),
    ("daysOfPayablesOutstanding", 65,"PEAD 22/40, XSec 14/25, Quality 14/15, Interp 7/10, NonRed 8/10"),
    ("receivablesTurnover", 65,   "PEAD 22/40, XSec 14/25, Quality 14/15, Interp 7/10, NonRed 8/10"),
    ("inventoryTurnover", 64,     "PEAD 22/40, XSec 14/25, Quality 14/15, Interp 7/10, NonRed 7/10"),
    ("assetTurnover", 63,         "PEAD 21/40, XSec 14/25, Quality 14/15, Interp 7/10, NonRed 7/10"),
    ("dividendYield", 61,         "PEAD 18/40, XSec 15/25, Quality 13/15, Interp 8/10, NonRed 7/10"),
    ("payoutRatio", 60,           "PEAD 17/40, XSec 15/25, Quality 13/15, Interp 8/10, NonRed 7/10"),
]

KM_FEATURES_DEFAULT = [c for (c, _, _) in sorted(KM_FEATURES_SCORED, key=lambda x: x[1], reverse=True)]
RT_FEATURES_DEFAULT = [c for (c, _, _) in sorted(RT_FEATURES_SCORED, key=lambda x: x[1], reverse=True)]


def _pick_existing_cols(df: pd.DataFrame, wanted: Iterable[str]) -> list[str]:
    have = set(df.columns)
    picked = [c for c in wanted if c in have]
    missing = [c for c in wanted if c not in have]
    if missing:
        print(f"[WARN] Missing columns (skipped): {missing}", flush=True)
    return picked


def load_returns(data_dir: Path, ticker: str, name: str) -> pd.DataFrame:
    p = data_dir / ticker / "events" / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing returns: {p} (run 23_compute_event_returns.py)")
    df = pd.read_csv(p)
    df["ticker"] = ticker
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def load_calendar(data_dir: Path, ticker: str) -> pd.DataFrame:
    p = data_dir / ticker / "calendar" / "earnings_calendar.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing calendar: {p}")
    df = pd.read_csv(p)
    df["ticker"] = ticker
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def load_key_metrics(data_dir: Path, ticker: str) -> pd.DataFrame:
    p = data_dir / ticker / "financials" / "key_metrics_quarter.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def load_ratios(data_dir: Path, ticker: str) -> pd.DataFrame:
    p = data_dir / ticker / "financials" / "ratios_quarter.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def _prefix(df: pd.DataFrame, prefix: str, keep: list[str]) -> pd.DataFrame:
    rename = {c: f"{prefix}{c}" for c in df.columns if c not in keep}
    return df.rename(columns=rename)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def _drop_symbol_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Robustly remove any symbol artifacts after merges
    drop_cols = [c for c in df.columns if c in ("symbol", "symbol_x", "symbol_y")]
    return df.drop(columns=drop_cols, errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--returns-name", default="event_returns")
    ap.add_argument("--out-name", default="event_panel_numeric")
    ap.add_argument("--km-cols", default=None)
    ap.add_argument("--rt-cols", default=None)
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    out = load_returns(data_dir, ticker, args.returns_name)

    # ---- Calendar surprises ----
    cal = load_calendar(data_dir, ticker)
    for col in ["eps_est", "eps_actual", "revenue_est", "revenue_actual"]:
        if col in cal.columns:
            cal[col] = pd.to_numeric(cal[col], errors="coerce")

    cal["eps_surprise_pct"] = _safe_div(cal["eps_actual"] - cal["eps_est"], cal["eps_est"].abs())
    cal["revenue_surprise_pct"] = _safe_div(cal["revenue_actual"] - cal["revenue_est"], cal["revenue_est"].abs())
    cal["eps_surprise_pct"] = 100 * cal["eps_surprise_pct"]          # in percent
    cal["revenue_surprise_pct"] = 100 * cal["revenue_surprise_pct"]  # in percent

    cal_keep_cols = _pick_existing_cols(
        cal,
        [
            "ticker",
            "earnings_date",
            # "announce_timing",
            "eps_est",
            "eps_actual",
            "revenue_est",
            "revenue_actual",
            "eps_surprise_pct",
            "revenue_surprise_pct",
        ],
    )
    out = out.merge(cal[cal_keep_cols], on=["ticker", "earnings_date"], how="left")

    # ---- Fundamentals ----
    km = load_key_metrics(data_dir, ticker)
    rt = load_ratios(data_dir, ticker)

    km_wanted = [c.strip() for c in args.km_cols.split(",") if c.strip()] if args.km_cols else KM_FEATURES_DEFAULT
    rt_wanted = [c.strip() for c in args.rt_cols.split(",") if c.strip()] if args.rt_cols else RT_FEATURES_DEFAULT

    if not km.empty:
        km_cols = _pick_existing_cols(km, ["symbol", "earnings_date"] + km_wanted)
        km2 = _prefix(km[km_cols].copy(), "km_", keep=["symbol", "earnings_date"])
        merged = out.merge(
            km2,
            left_on=["ticker", "earnings_date"],
            right_on=["symbol", "earnings_date"],
            how="left",
        )
        out = _drop_symbol_cols(merged)

    if not rt.empty:
        rt_cols = _pick_existing_cols(rt, ["symbol", "earnings_date"] + rt_wanted)
        rt2 = _prefix(rt[rt_cols].copy(), "rt_", keep=["symbol", "earnings_date"])
        merged = out.merge(
            rt2,
            left_on=["ticker", "earnings_date"],
            right_on=["symbol", "earnings_date"],
            how="left",
        )
        out = _drop_symbol_cols(merged)

    # ---- Fill-forward fundamentals to eliminate "pointless" missingness ----
    # Quarterly fundamentals should behave like step functions between reports.
    out["_earn_dt"] = pd.to_datetime(out["earnings_date"], errors="coerce")
    out = out.sort_values("_earn_dt")

    fund_cols = [c for c in out.columns if c.startswith(("km_", "rt_"))]
    if fund_cols:
        out[fund_cols] = out[fund_cols].apply(pd.to_numeric, errors="coerce").ffill()

    out = out.drop(columns=["_earn_dt"])

    out_dir = data_dir / ticker / "events"

    ensure_dir(out_dir)
    out_csv = out_dir / f"{args.out_name}.csv"
    out.to_csv(out_csv, index=False)

    meta = {
        "ticker": ticker,
        "rows": int(out.shape[0]),
        "created_at_local": datetime.now().isoformat(),
        "km_cols_requested": km_wanted,
        "rt_cols_requested": rt_wanted,
        "km_scored": KM_FEATURES_SCORED,
        "rt_scored": RT_FEATURES_SCORED,
        "notes": [
            "Merged event_returns + calendar surprises (surprises stored as percent).",
            "Merged selected key_metrics as km_* and selected ratios as rt_*.",
            "Robustly drops symbol / symbol_x / symbol_y after merges.",
        ],
    }
    (out_dir / f"{args.out_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] {ticker}: wrote {out_csv} ({out.shape[0]} rows)")


if __name__ == "__main__":
    main()