#!/usr/bin/env python3
# scripts/30_fit_event_feature_stats.py

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# Columns that are not features (identifiers/targets)
DROP_FEATURES = {
    "ticker",
    "earnings_date",
    "window",
    "y_type",
    "y",
    # upstream merge metadata
    "announce_timing",
    "pre_close_date",
    "day0_date",
    "dt_m5",
    "dt_m1",
    "dt_p1",
    "dt_p5",
    "dt_p10",
    "path_start",
    "path_end",
    "pre_bdays",
    "post_bdays",
}

# By default we EXCLUDE price-path / window-return columns from "features"
# because they leak post-event information into correlations.
LEAKY_PREFIXES_DEFAULT = (
    "px_",
    "mkt_px_",
    "logpct_",     # your biggest offender (often same info as y)
    "pct_",        # y is built from pct_*, but keep exclusion anyway
    "ar_",         # abnormal-return path columns (if any)
    "car_",        # cumulative abnormal return columns (targets)
    "ret_",        # generic returns (if you stored them)
)

# Allow-list: if present, we prefer these families as "features"
ALLOW_PREFIXES_DEFAULT = ("sur_", "est_", "gro_", "km_", "ratio_", "alpha_", "beta_", "sigma_", "vol_", "tech_")


def _as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pearson(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    """
    Pearson correlation (pooled). Returns (corr, n_effective)
    """
    x = _as_num(x)
    y = _as_num(y)
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 3:
        return np.nan, n

    xx = x[m].to_numpy(dtype=float)
    yy = y[m].to_numpy(dtype=float)

    sx = float(np.std(xx))
    sy = float(np.std(yy))
    if sx < 1e-12 or sy < 1e-12:
        return np.nan, n

    return float(np.corrcoef(xx, yy)[0, 1]), n


def _is_leaky_col(c: str, leaky_prefixes: tuple[str, ...]) -> bool:
    return c.startswith(leaky_prefixes)


def infer_feature_cols(
    df: pd.DataFrame,
    allow_prefixes: tuple[str, ...],
    leaky_prefixes: tuple[str, ...],
    include_leaky: bool,
) -> List[str]:
    """
    Infer numeric feature columns.

    Logic:
      - drop identifiers/targets in DROP_FEATURES
      - optionally drop leaky prefixes
      - keep numeric or numeric-like columns
      - if allow_prefixes exist in the df, restrict to those families (more robust)
    """
    cols = [c for c in df.columns if c not in DROP_FEATURES]

    if not include_leaky:
        cols = [c for c in cols if not _is_leaky_col(c, leaky_prefixes)]

    # numeric-like
    numeric_candidates = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric_candidates.append(c)
        else:
            # numeric-like object?
            ss = pd.to_numeric(s, errors="coerce")
            if ss.notna().sum() >= max(10, int(0.05 * len(ss))):
                numeric_candidates.append(c)

    # If any allowed prefix families exist, restrict to those (prevents accidental metadata leakage)
    has_allowed = any(any(c.startswith(p) for p in allow_prefixes) for c in numeric_candidates)
    if has_allowed:
        numeric_candidates = [c for c in numeric_candidates if any(c.startswith(p) for p in allow_prefixes)]

    return numeric_candidates


def print_top(df: pd.DataFrame, top: int) -> None:
    if df.empty:
        print("[WARN] No correlation rows to print.")
        return

    for (ytype, win), g in df.groupby(["y_type", "window"], sort=True):
        g2 = g.sort_values("abs_corr", ascending=False).head(top)
        print(f"\n=== y_type={ytype} | window={win} | top={min(top, len(g2))} ===")
        for _, r in g2.iterrows():
            corr = r["corr"]
            n = int(r["n"])
            feat = r["feature"]
            if pd.isna(corr):
                print(f"  {feat:40s}  corr=NaN   n={n}")
            else:
                print(f"  {feat:40s}  corr={corr:+.4f}  |corr|={abs(corr):.4f}  n={n}")


def _print_note(df: pd.DataFrame, expected_per_ticker: int) -> None:
    # counts are at the EVENT level (unique ticker+earnings_date)
    ev = df[["ticker", "earnings_date"]].drop_duplicates()
    nt = int(ev["ticker"].nunique())
    ne = int(len(ev))
    per = ev.groupby("ticker").size().sort_values()
    miss = nt * int(expected_per_ticker) - ne

    print("\n[NOTE] Event coverage diagnostic")
    print(f"[NOTE] tickers={nt} unique_events={ne} expected_per_ticker={expected_per_ticker} -> missing_events={miss}")

    if len(per) > 0:
        print(f"[NOTE] events/ticker: min={int(per.min())} median={int(per.median())} max={int(per.max())}")
        bad = per[per < expected_per_ticker]
        if len(bad) > 0:
            # print up to 10 tickers with short calendars
            show = bad.head(10)
            print("[NOTE] tickers with < expected events (showing up to 10): " +
                  ", ".join([f"{t}={int(n)}" for t, n in show.items()]))

    # show how many y's exist per (y_type, window)
    if "y" in df.columns:
        ycov = df.groupby(["y_type", "window"])["y"].apply(lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum()))
        # print compact
        for (yt, w), n in ycov.items():
            print(f"[NOTE] y non-null: y_type={yt} window={w} n={n}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pooled feature correlations vs event-window target (sorted by abs corr).")
    ap.add_argument("--long", default="data/_derived/event_long.csv")
    ap.add_argument("--out-dir", default="data/_derived/fits")
    ap.add_argument(
        "--y-type",
        default="car_mm_1y_pct",
        help="Target: ret_pct, car_simple_pct, car_mm_1y_pct, car_mm_1q_pct, or 'all'. Default: car_mm_1y_pct",
    )
    ap.add_argument("--windows", nargs="*", default=None, help="Subset of windows. Default: all.")
    ap.add_argument("--min-n", type=int, default=30, help="Minimum paired observations for a correlation to be kept.")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--round", type=int, default=4)

    # NEW: diagnostics + leakage controls
    ap.add_argument("--expected-per-ticker", type=int, default=20)
    ap.add_argument("--include-price-derived", action="store_true", help="If set, include px_/logpct_ etc as features (NOT recommended).")

    args = ap.parse_args()

    long_path = Path(args.long)
    if not long_path.exists():
        raise FileNotFoundError(f"Missing long dataset: {long_path} (run export script first)")

    df = pd.read_csv(long_path)

    # Coverage note (explains 319/318-type behavior)
    _print_note(df, expected_per_ticker=int(args.expected_per_ticker))

    # Select y_type(s)
    if str(args.y_type).lower() == "all":
        y_types = sorted(df["y_type"].dropna().unique().tolist())
    else:
        y_types = [args.y_type]
        df = df[df["y_type"] == args.y_type].copy()

    # Select windows
    if args.windows:
        want = set(args.windows)
        df = df[df["window"].isin(want)].copy()

    allow_prefixes = ALLOW_PREFIXES_DEFAULT
    leaky_prefixes = LEAKY_PREFIXES_DEFAULT

    # Identify feature columns
    feat_cols = infer_feature_cols(
        df,
        allow_prefixes=allow_prefixes,
        leaky_prefixes=leaky_prefixes,
        include_leaky=bool(args.include_price_derived),
    )
    if not feat_cols:
        raise RuntimeError(
            "No feature columns found after filtering. "
            "If your fundamentals don’t use prefixes (sur_/est_/gro_), either rename them or relax allowlist logic."
        )

    rows = []
    for ytype in y_types:
        sub_y = df[df["y_type"] == ytype] if len(y_types) > 1 else df
        for win, g in sub_y.groupby("window", sort=True):
            for feat in feat_cols:
                corr, n = _pearson(g[feat], g["y"])
                if n < int(args.min_n):
                    continue
                rows.append({"y_type": ytype, "window": win, "feature": feat, "corr": corr, "abs_corr": abs(corr) if np.isfinite(corr) else np.nan, "n": n})

    out = pd.DataFrame(rows)
    if out.empty:
        print("[WARN] No correlations met min-n; try lowering --min-n.")
        return

    out["corr"] = pd.to_numeric(out["corr"], errors="coerce").round(int(args.round))
    out["abs_corr"] = pd.to_numeric(out["abs_corr"], errors="coerce").round(int(args.round))

    out = out.sort_values(["y_type", "window", "abs_corr"], ascending=[True, True, False]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "pooled_corrs.csv"
    out.to_csv(out_csv, index=False)

    print(f"\n[OK] wrote pooled correlations -> {out_csv}")
    if not args.include_price_derived:
        print("[INFO] price-path columns (px_/logpct_/etc) excluded from feature set to prevent leakage.")
    print_top(out, top=int(args.top))


if __name__ == "__main__":
    main()
