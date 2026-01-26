#!/usr/bin/env python3
# scripts/32_viz_surprise_vs_drift.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import default_data_dir, ensure_dir, now_iso


def _coalesce_timing(df: pd.DataFrame) -> pd.Series:
    if "announce_timing" in df.columns:
        return df["announce_timing"]
    x = df["announce_timing_x"] if "announce_timing_x" in df.columns else pd.Series([np.nan] * len(df))
    y = df["announce_timing_y"] if "announce_timing_y" in df.columns else pd.Series([np.nan] * len(df))
    return x.combine_first(y)


def _read_panel(data_dir: Path, ticker: str, panel_name: str) -> pd.DataFrame:
    p = data_dir / ticker / "events" / f"{panel_name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing panel: {p}")
    df = pd.read_csv(p)
    df["announce_timing"] = _coalesce_timing(df)
    return df


def _savefig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _scatter_with_fit(df: pd.DataFrame, xcol: str, ycol: str, title: str, out_path: Path, dpi: int) -> None:
    x = pd.to_numeric(df.get(xcol), errors="coerce")
    y = pd.to_numeric(df.get(ycol), errors="coerce")
    m = (~x.isna()) & (~y.isna())
    x, y = x[m], y[m]
    if len(x) < 3:
        return

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, alpha=0.8)
    ax.axhline(0, linewidth=1, alpha=0.35)
    ax.axvline(0, linewidth=1, alpha=0.35)

    # simple fit
    beta = np.polyfit(x.values, y.values, deg=1)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(xs, beta[0] * xs + beta[1], linewidth=2, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)
    _savefig(fig, out_path, dpi)


def main() -> None:
    ap = argparse.ArgumentParser(description="Scatter: earnings surprise vs abnormal drift.")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--panel-name", default="event_panel")
    ap.add_argument("--out-subdir", default="viz")
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    df = _read_panel(data_dir, ticker, args.panel_name)

    out_dir = data_dir / ticker / args.out_subdir
    ensure_dir(out_dir)

    # Core PEAD narrative: surprise -> drift (post-reaction)
    _scatter_with_fit(
        df,
        "eps_surprise_pct",
        "abn_retpct_drift_20_vs_react",
        "EPS surprise vs 20d abnormal drift (REACT→+20)",
        out_dir / f"{ticker}_05_scatter_eps_surprise_vs_abn_drift20.png",
        dpi=args.dpi,
    )

    _scatter_with_fit(
        df,
        "revenue_surprise_pct",
        "abn_retpct_drift_20_vs_react",
        "Revenue surprise vs 20d abnormal drift (REACT→+20)",
        out_dir / f"{ticker}_06_scatter_rev_surprise_vs_abn_drift20.png",
        dpi=args.dpi,
    )

    (out_dir / f"{ticker}_32_viz_surprise_vs_drift.meta.txt").write_text(
        f"ticker={ticker}\ncreated_at={now_iso()}\npanel={args.panel_name}\nrows={len(df)}\n",
        encoding="utf-8",
    )
    print(f"[OK] {ticker}: wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
