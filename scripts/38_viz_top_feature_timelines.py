#!/usr/bin/env python3
# scripts/38_viz_top_feature_timelines.py

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
    df = pd.read_csv(p)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    return df


def _savefig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    m = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return x * np.nan
    return (x - m) / sd


def main() -> None:
    ap = argparse.ArgumentParser(description="Timeline of top predictors (z-scored) ordered by earnings date.")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--panel-name", default="event_panel")
    ap.add_argument("--out-subdir", default="viz")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--target", default="abn_retpct_drift_20_vs_react")
    ap.add_argument("--include-target", action="store_true")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    df = _read_panel(data_dir, ticker, args.panel_name).sort_values("earnings_date")

    out_dir = data_dir / ticker / args.out_subdir
    ensure_dir(out_dir)

    rank_path = out_dir / f"{ticker}_09_corr_rank_top_predictors.csv"
    features: list[str] = []
    if rank_path.exists():
        try:
            rank = pd.read_csv(rank_path, index_col=0)
            features = rank.index.tolist()
        except Exception:
            features = []

    if not features:
        num = df.select_dtypes(include=[np.number]).copy()
        if args.target not in num.columns or num[args.target].dropna().empty:
            print(f"[WARN] {ticker}: target column missing/empty; skipping.")
            return
        tgt = num[args.target]
        corrs = {}
        for c in num.columns:
            if c == args.target:
                continue
            v = num[c]
            m = (~tgt.isna()) & (~v.isna())
            if m.sum() < 5:
                continue
            corrs[c] = float(np.corrcoef(tgt[m], v[m])[0, 1])
        if not corrs:
            print(f"[WARN] {ticker}: not enough numeric features for correlation; skipping.")
            return
        features = [k for k, _ in sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)]

    wanted = []
    for c in features:
        if c in df.columns:
            wanted.append(c)
        if len(wanted) >= args.top_k:
            break
    if not wanted:
        print(f"[WARN] {ticker}: no ranked predictors present in panel; skipping.")
        return

    ts = df[["earnings_date"]].copy()
    for c in wanted:
        ts[c] = _zscore(df[c])

    if args.include_target and args.target in df.columns:
        ts[args.target] = _zscore(df[args.target])

    ts = ts.dropna(subset=["earnings_date"])
    if ts.empty:
        return

    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(1, 1, 1)

    for c in wanted:
        ax.plot(ts["earnings_date"], ts[c], marker="o", linewidth=2, alpha=0.85, label=c)

    if args.include_target and args.target in ts.columns:
        ax.plot(ts["earnings_date"], ts[args.target], marker="o", linewidth=3, alpha=0.9, label=args.target)

    ax.axhline(0, linewidth=1, alpha=0.35)
    ax.set_title(f"Top predictors over time (z-scored, date order) — {ticker}")
    ax.set_xlabel("Earnings date")
    ax.set_ylabel("z-score")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2, loc="best")

    _savefig(fig, out_dir / f"{ticker}_15_top_feature_zscore_timeline.png", dpi=args.dpi)

    (out_dir / f"{ticker}_38_viz_top_feature_timelines.meta.txt").write_text(
        f"ticker={ticker}\ncreated_at={now_iso()}\npanel={args.panel_name}\n"
        f"top_k={args.top_k}\ntarget={args.target}\nfeatures={wanted}\n",
        encoding="utf-8",
    )
    print(f"[OK] {ticker}: wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
