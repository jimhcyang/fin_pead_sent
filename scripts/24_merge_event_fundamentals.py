#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/24_merge_event_fundamentals.py (v3.1)

Merges:
  events/event_study_panel.csv
with:
  financials/stable_surprises_{period}(.tailN).csv
  financials/stable_income_statement_{period}.csv
  financials/stable_income_statement_growth_{period}.csv
  financials/stable_analyst_estimates_{period}.csv

Key:
- Matches each earnings event to the latest fundamental period_end <= event_date
  via merge_asof(direction="backward").

Fixes your crash:
- stable_analyst_estimates often has date_x/date_y; we treat date_x as the period end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return pd.read_csv(p)


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _detect_period_end_col(df: pd.DataFrame) -> str:
    # priority order
    for c in ("period_end", "date", "date_x"):
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a period_end/date column. Columns={list(df.columns)[:50]}")


def _normalize_stable(df: pd.DataFrame, src_name: str, prefix: str) -> pd.DataFrame:
    """
    Produces:
      {prefix}period_end_dt  (datetime64)  <- merge key
      {prefix}period_end     (YYYY-MM-DD string)
    and prefixes all other columns to prevent collisions.
    """
    df = df.copy()

    # analyst estimates: date_x = anchor (period end), date_y = raw API date
    if "date_x" in df.columns and "date_y" in df.columns and src_name.startswith("stable_analyst_estimates"):
        df = df.rename(columns={"date_y": "estimate_date_raw"})

    pe_col = _detect_period_end_col(df)
    df["_pe_dt"] = _to_dt(df[pe_col])

    df = df.dropna(subset=["_pe_dt"]).sort_values("_pe_dt").reset_index(drop=True)
    df[f"{prefix}period_end_dt"] = df["_pe_dt"]
    df[f"{prefix}period_end"] = df["_pe_dt"].dt.strftime("%Y-%m-%d")

    keep = {f"{prefix}period_end_dt", f"{prefix}period_end"}

    rename = {}
    for c in df.columns:
        if c in keep or c == "_pe_dt":
            continue
        rename[c] = f"{prefix}{c}"
    df = df.rename(columns=rename)

    df = df.drop(columns=["_pe_dt"], errors="ignore")
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df


def _asof_merge(ev: pd.DataFrame, stable: pd.DataFrame, right_dt_col: str) -> pd.DataFrame:
    left = ev.sort_values("event_date_dt").reset_index(drop=True)
    right = stable.sort_values(right_dt_col).reset_index(drop=True)

    return pd.merge_asof(
        left,
        right,
        left_on="event_date_dt",
        right_on=right_dt_col,
        direction="backward",
        allow_exact_matches=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--period", default="quarter", choices=["quarter", "annual"])
    ap.add_argument("--out-name", default="event_ml_panel")
    args = ap.parse_args()

    tkr = args.ticker.upper()
    tdir = args.data_dir / tkr

    ev_path = tdir / "events" / "event_study_panel.csv"
    ev = _load_csv(ev_path)

    if "event_date" not in ev.columns:
        raise ValueError(f"{ev_path} missing required column: event_date")

    ev = ev.copy()
    ev["event_date_dt"] = _to_dt(ev["event_date"])
    ev = ev.dropna(subset=["event_date_dt"]).sort_values("event_date_dt").reset_index(drop=True)

    fin = tdir / "financials"

    # Prefer tailN surprises if present
    sur_candidates = sorted(fin.glob(f"stable_surprises_{args.period}.tail*.csv"))
    sur_path = sur_candidates[-1] if sur_candidates else (fin / f"stable_surprises_{args.period}.csv")

    inc_path = fin / f"stable_income_statement_{args.period}.csv"
    gro_path = fin / f"stable_income_statement_growth_{args.period}.csv"
    est_path = fin / f"stable_analyst_estimates_{args.period}.csv"

    sur = _normalize_stable(_load_csv(sur_path), f"stable_surprises_{args.period}", "sur_")
    inc = _normalize_stable(_load_csv(inc_path), f"stable_income_statement_{args.period}", "inc_")
    gro = _normalize_stable(_load_csv(gro_path), f"stable_income_statement_growth_{args.period}", "gro_")
    est = _normalize_stable(_load_csv(est_path), f"stable_analyst_estimates_{args.period}", "est_")

    out = _asof_merge(ev, sur, "sur_period_end_dt")
    out = _asof_merge(out, inc, "inc_period_end_dt")
    out = _asof_merge(out, gro, "gro_period_end_dt")
    out = _asof_merge(out, est, "est_period_end_dt")

    out_dir = tdir / "events"
    ensure_dir(out_dir)

    out_path = out_dir / f"{args.out_name}.csv"
    out.to_csv(out_path, index=False)

    tgt = "drift_car_mm_pct"
    nonmiss = int(pd.to_numeric(out.get(tgt, pd.Series(dtype=float)), errors="coerce").notna().sum()) if tgt in out.columns else 0
    print(f"[OK] {tkr}: wrote {out_path} | rows={len(out)} | target_nonmissing({tgt})={nonmiss}")


if __name__ == "__main__":
    main()
