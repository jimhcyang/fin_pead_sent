#!/usr/bin/env python3
# scripts/41_text_tone_vs_car_heatmap.py
"""
Compute correlation matrix between *LM tone* variants and CAR variants, and render a heatmap.

Why this exists:
- Your merged panel (event_text_features.csv) contains many columns with names like
  'tr__tr_lm_tone__qa__all' and 'news__nw_lm_tone__post'.
- In prior code, suffix-based feature detection missed these and correlations were skipped.

Outputs (default, under --out-dir):
- corr_text_tone_vs_car.csv
- corr_text_tone_vs_car.png

Usage:
  python scripts/41_text_tone_vs_car_heatmap.py --panel data/_derived/text/event_text_features.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pick_first(cols: List[str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        r = re.compile(pat, re.I)
        for c in cols:
            if r.search(c):
                return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=str, default="data/_derived/text/event_text_features.csv")
    ap.add_argument("--out-dir", type=str, default="data/_derived/text")
    ap.add_argument("--min-n", type=int, default=80, help="Min paired observations per corr cell.")
    args = ap.parse_args()

    panel_path = Path(args.panel)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_path)

    cols = list(df.columns)

    # Robustly locate tone columns (handles your current naming)
    tr_prepared_col = _pick_first(cols, [r"^tr__.*lm_tone.*prepared", r"^tr__tr_lm_tone__prepared"])
    tr_qa_col       = _pick_first(cols, [r"^tr__.*lm_tone.*qa",       r"^tr__tr_lm_tone__qa"])
    news_pre_col    = _pick_first(cols, [r"^news__.*lm_tone.*pre",    r"^news__nw_lm_tone__pre"])
    news_post_col   = _pick_first(cols, [r"^news__.*lm_tone.*post",   r"^news__nw_lm_tone__post"])

    missing = [k for k,v in {
        "tr_prepared": tr_prepared_col, "tr_qa": tr_qa_col,
        "news_pre": news_pre_col, "news_post": news_post_col,
    }.items() if v is None]
    if missing:
        raise RuntimeError(f"Could not find required tone columns in panel: {missing}")

    tone = pd.DataFrame({
        "tr_prepared": pd.to_numeric(df[tr_prepared_col], errors="coerce"),
        "tr_qa": pd.to_numeric(df[tr_qa_col], errors="coerce"),
        "news_pre": pd.to_numeric(df[news_pre_col], errors="coerce"),
        "news_post": pd.to_numeric(df[news_post_col], errors="coerce"),
    })
    tone["tr_avg"] = (tone["tr_prepared"] + tone["tr_qa"]) / 2
    tone["tr_diff"] = tone["tr_qa"] - tone["tr_prepared"]
    tone["news_avg"] = (tone["news_pre"] + tone["news_post"]) / 2
    tone["news_diff"] = tone["news_post"] - tone["news_pre"]

    tone = tone[["tr_prepared","tr_qa","tr_avg","tr_diff","news_pre","news_post","news_avg","news_diff"]]

    # CAR columns
    car_cols = [c for c in cols if str(c).lower().startswith("car_")]
    if not car_cols:
        raise RuntimeError("No CAR columns found (expected columns starting with 'car_').")

    # Filter out CAR columns with no coverage (e.g., short windows if CAR calc was NaN)
    min_n = int(args.min_n)
    car_nonnull = {c: pd.to_numeric(df[c], errors="coerce").notna().sum() for c in car_cols}
    car_keep = sorted([c for c,n in car_nonnull.items() if n >= min_n])
    if not car_keep:
        raise RuntimeError(f"No CAR columns have >= {min_n} non-missing observations.")

    corr = pd.DataFrame(index=tone.columns, columns=car_keep, dtype=float)
    nobs = pd.DataFrame(index=tone.columns, columns=car_keep, dtype=int)

    for f in tone.columns:
        x = pd.to_numeric(tone[f], errors="coerce")
        for ycol in car_keep:
            y = pd.to_numeric(df[ycol], errors="coerce")
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            nobs.loc[f, ycol] = n
            corr.loc[f, ycol] = float(np.corrcoef(x[mask], y[mask])[0, 1]) if n >= min_n else np.nan

    out_csv = out_dir / "corr_text_tone_vs_car.csv"
    corr.to_csv(out_csv, index=True)

    # Heatmap (matplotlib defaults; no explicit colors)
    fig = plt.figure(figsize=(max(10, 0.35*len(car_keep)+4), 6))
    ax = plt.gca()
    im = ax.imshow(corr.values, aspect="auto")

    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(list(corr.index))
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(list(corr.columns), rotation=60, ha="right")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title("Correlation: LM tone variants vs CAR variants (pairwise, Pearson)")
    fig.tight_layout()

    out_png = out_dir / "corr_text_tone_vs_car.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print("[OK] wrote:", out_csv)
    print("[OK] wrote:", out_png)
    print("[INFO] tone cols used:", {"tr_prepared": tr_prepared_col, "tr_qa": tr_qa_col, "news_pre": news_pre_col, "news_post": news_post_col})
    print("[INFO] kept CAR cols:", len(car_keep), "of", len(car_cols))


if __name__ == "__main__":
    main()
