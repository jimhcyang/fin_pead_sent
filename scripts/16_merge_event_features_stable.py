#!/usr/bin/env python3
# scripts/16_merge_event_features_stable.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _eventlib import default_data_dir, ensure_dir, save_with_meta


def _load_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return pd.read_csv(p)


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _detect_period_end_col(df: pd.DataFrame) -> str:
    for c in ("period_end", "date", "date_x"):
        if c in df.columns:
            return c
    raise ValueError(f"Could not find period_end/date column. cols={list(df.columns)[:50]}")


def _normalize(df: pd.DataFrame, src_name: str, prefix: str) -> pd.DataFrame:
    df = df.copy()
    # analyst estimates sometimes has date_x/date_y
    if "date_x" in df.columns and "date_y" in df.columns and src_name.startswith("stable_analyst_estimates"):
        df = df.rename(columns={"date_y": "estimate_date_raw"})

    pe_col = _detect_period_end_col(df)
    df["_pe_dt"] = _to_dt(df[pe_col])
    df = df.dropna(subset=["_pe_dt"]).sort_values("_pe_dt").reset_index(drop=True)
    df[f"{prefix}period_end_dt"] = df["_pe_dt"]
    df[f"{prefix}period_end"] = df["_pe_dt"].dt.strftime("%Y-%m-%d")

    keep = {f"{prefix}period_end_dt", f"{prefix}period_end"}

    ren = {}
    for c in df.columns:
        if c in keep or c == "_pe_dt":
            continue
        ren[c] = f"{prefix}{c}"
    df = df.rename(columns=ren).drop(columns=["_pe_dt"], errors="ignore")
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df


def _asof_merge(ev: pd.DataFrame, stable: pd.DataFrame, right_dt_col: str) -> pd.DataFrame:
    left = ev.sort_values("event_date_dt").reset_index(drop=True)
    right = stable.sort_values(right_dt_col).reset_index(drop=True)
    return pd.merge_asof(left, right, left_on="event_date_dt", right_on=right_dt_col, direction="backward", allow_exact_matches=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--period", choices=["quarter", "annual"], default="quarter")
    ap.add_argument("--windows-name", default="event_windows")
    ap.add_argument("--out-name", default="event_features_stable")
    args = ap.parse_args()

    tkr = args.ticker.upper()
    tdir = args.data_dir / tkr
    fin = tdir / "financials"

    ev_path = tdir / "events" / f"{args.windows_name}.csv"
    if not ev_path.exists():
        raise FileNotFoundError(f"Missing windows file: {ev_path} (run script 11)")
    ev = pd.read_csv(ev_path)
    ev["event_date_dt"] = pd.to_datetime(ev["earnings_date"], errors="coerce")
    ev = ev.dropna(subset=["event_date_dt"]).sort_values("event_date_dt").reset_index(drop=True)

    # prefer tailN surprises if present
    sur_candidates = sorted(fin.glob(f"stable_surprises_{args.period}.tail*.csv"))
    sur_path = sur_candidates[-1] if sur_candidates else (fin / f"stable_surprises_{args.period}.csv")

    incg_path = fin / f"stable_income_statement_growth_{args.period}.csv"
    est_path = fin / f"stable_analyst_estimates_{args.period}.csv"

    sur = _normalize(_load_csv(sur_path), f"stable_surprises_{args.period}", "sur_")
    gro = _normalize(_load_csv(incg_path), f"stable_income_statement_growth_{args.period}", "gro_")
    est = _normalize(_load_csv(est_path), f"stable_analyst_estimates_{args.period}", "est_")

    out = _asof_merge(ev, sur, "sur_period_end_dt")
    out = _asof_merge(out, gro, "gro_period_end_dt")
    out = _asof_merge(out, est, "est_period_end_dt")

    out_dir = tdir / "events"
    ensure_dir(out_dir)
    out_csv = out_dir / f"{args.out_name}.csv"

    save_with_meta(
        out,
        out_csv,
        meta={
            "ticker": tkr,
            "notes": [
                "Stable fundamentals merged as-of (backward) to earnings_date.",
                "Prefixes: sur_ (surprises), gro_ (income_statement_growth), est_ (analyst_estimates).",
            ],
        },
    )

    print(f"[OK] {tkr}: wrote {out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
