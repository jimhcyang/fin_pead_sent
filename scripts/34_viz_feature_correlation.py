#!/usr/bin/env python3
# scripts/34_viz_feature_correlation.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import default_data_dir, ensure_dir, now_iso


def _read_panel(data_dir: Path, ticker: str, panel_name: str) -> pd.DataFrame:
    p = data_dir / ticker / "events" / f"{panel_name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing panel: {p}")
    return pd.read_csv(p)


def _savefig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Correlation heatmap: top predictors vs abnormal drift.")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--panel-name", default="event_panel")  # includes abn_ + km_/rt_
    ap.add_argument("--out-subdir", default="viz")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--target", default="abn_retpct_drift_20_vs_react")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-missing", type=float, default=0.50)  # drop cols with >50% missing
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    df = _read_panel(data_dir, ticker, args.panel_name).copy()

    out_dir = data_dir / ticker / args.out_subdir
    ensure_dir(out_dir)

    if args.target not in df.columns:
        print(f"[WARN] {ticker}: target {args.target} missing; skipping.")
        return

    # candidate predictors: surprises + km_* + rt_* (+ optionally returns)
    candidates = []
    for c in df.columns:
        if c in {"eps_surprise_pct", "revenue_surprise_pct"}:
            candidates.append(c)
        elif c.startswith("km_") or c.startswith("rt_"):
            candidates.append(c)

    # numeric only + missingness filter
    X = df[candidates].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[args.target], errors="coerce")
    miss = X.isna().mean()
    keep = miss[miss <= args.max_missing].index.tolist()
    X = X[keep]

    if X.shape[1] == 0:
        print(f"[WARN] {ticker}: no usable predictors after missingness filter.")
        return

    # pick top-k by abs correlation with target
    corr_with_target = X.corrwith(y).dropna().abs().sort_values(ascending=False)
    top = corr_with_target.head(args.topk).index.tolist()
    cols = [args.target] + top

    M = pd.concat([y.rename(args.target), X[top]], axis=1).corr()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(M.values, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title(f"Correlation heatmap (top {len(top)} predictors vs {args.target})")

    _savefig(fig, out_dir / f"{ticker}_09_corr_heatmap_top_predictors.png", dpi=args.dpi)

    # also dump a ranked list for narrative writing
    rank_path = out_dir / f"{ticker}_09_corr_rank_top_predictors.csv"
    corr_with_target.rename("abs_corr_with_target").to_csv(rank_path)

    (out_dir / f"{ticker}_34_viz_feature_correlation.meta.txt").write_text(
        f"ticker={ticker}\ncreated_at={now_iso()}\npanel={args.panel_name}\ntarget={args.target}\n"
        f"topk={args.topk}\nmax_missing={args.max_missing}\n",
        encoding="utf-8",
    )
    print(f"[OK] {ticker}: wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
