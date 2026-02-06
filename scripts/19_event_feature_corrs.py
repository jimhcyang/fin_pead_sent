#!/usr/bin/env python3
# scripts/19_event_feature_corrs.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from _eventlib import ensure_dir


DROP_FEATURES = {
    "ticker",
    "earnings_date",
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
    "window",
    "y_type",
    "y",
}


def _as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pearson(x: pd.Series, y: pd.Series) -> float:
    x = _as_num(x)
    y = _as_num(y)
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return np.nan
    xx = x[m].to_numpy(dtype=float)
    yy = y[m].to_numpy(dtype=float)
    sx = np.std(xx)
    sy = np.std(yy)
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", default="data/_derived/event_long.csv")
    ap.add_argument("--out-dir", default="data/_derived/feature_corrs")
    ap.add_argument("--round", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.long)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    feat_cols = [c for c in df.columns if c not in DROP_FEATURES and df[c].dtype != "object"]

    if not feat_cols:
        raise RuntimeError("No numeric feature columns found (after drops).")

    pooled_rows = []
    by_ticker_rows = []

    for (ytype, win), sub in df.groupby(["y_type", "window"], sort=True):
        # pooled
        for feat in feat_cols:
            c = _pearson(sub[feat], sub["y"])
            pooled_rows.append(
                {"y_type": ytype, "window": win, "feature": feat, "corr": c, "abs_corr": abs(c) if np.isfinite(c) else np.nan}
            )

        # by ticker
        for tkr, g in sub.groupby("ticker", sort=True):
            for feat in feat_cols:
                c = _pearson(g[feat], g["y"])
                by_ticker_rows.append(
                    {"y_type": ytype, "window": win, "ticker": tkr, "feature": feat, "corr": c, "abs_corr": abs(c) if np.isfinite(c) else np.nan}
                )

    pooled = pd.DataFrame(pooled_rows)
    by_ticker = pd.DataFrame(by_ticker_rows)

    # company mean abs
    mean_abs = (
        by_ticker.groupby(["y_type", "window", "feature"], sort=True)["abs_corr"]
        .mean()
        .reset_index()
        .rename(columns={"abs_corr": "company_mean_abs"})
    )

    for t in [pooled, by_ticker, mean_abs]:
        for c in ["corr", "abs_corr", "company_mean_abs"]:
            if c in t.columns:
                t[c] = pd.to_numeric(t[c], errors="coerce").round(int(args.round))

    pooled.to_csv(out_dir / "pooled.csv", index=False)
    by_ticker.to_csv(out_dir / "by_ticker.csv", index=False)
    mean_abs.to_csv(out_dir / "company_mean_abs.csv", index=False)

    print(f"[OK] wrote: {out_dir / 'pooled.csv'}")
    print(f"[OK] wrote: {out_dir / 'by_ticker.csv'}")
    print(f"[OK] wrote: {out_dir / 'company_mean_abs.csv'}")


if __name__ == "__main__":
    main()
