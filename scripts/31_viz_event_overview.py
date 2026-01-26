#!/usr/bin/env python3
# scripts/31_viz_event_overview.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
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
        raise FileNotFoundError(f"Missing panel: {p} (run scripts/30_build_events_all.py first)")
    df = pd.read_csv(p)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    df["announce_timing"] = _coalesce_timing(df)
    return df


def _read_prices(data_dir: Path, ticker: str) -> Optional[pd.DataFrame]:
    p = data_dir / ticker / "prices" / "yf_ohlcv_daily.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # prefer adj close
    adj = None
    for c in ["adj_close", "Adj Close", "adjclose", "adjClose"]:
        if c in df.columns:
            adj = c
            break
    if adj is None:
        return None
    df = df[["date", adj]].rename(columns={adj: "adj_close"}).dropna()
    return df


def _savefig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _mean_and_ci(x: pd.Series) -> tuple[float, float]:
    """Mean and ~95% CI half-width using normal approx."""
    v = pd.to_numeric(x, errors="coerce").dropna()
    if len(v) <= 1:
        return (float(v.mean()) if len(v) else np.nan, np.nan)
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    return m, 1.96 * se


def plot_price_with_earnings_markers(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    # Color earnings markers by EPS surprise sign if available
    eps = pd.to_numeric(events.get("eps_surprise_pct", pd.Series([np.nan] * len(events))), errors="coerce")
    edates = pd.to_datetime(events["earnings_date"], errors="coerce")
    good = eps >= 0

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(prices["date"], prices["adj_close"])
    ax.set_title("Adjusted close with earnings markers")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adj close")

    # vertical lines
    for d, is_good in zip(edates, good):
        if pd.isna(d):
            continue
        ax.axvline(d, linewidth=1, alpha=0.25)

    ax.grid(True, alpha=0.25)
    _savefig(fig, out_path, dpi)


def plot_mean_abnormal_event_curve(
    panel: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    # “Event-time” points: pre->react is day+1, then +5/+10/+20 trading days from react
    # We plot cumulative abnormal returns from PRE (BHAR-style, market-adjusted).
    needed = [
        "abn_retpct_react_vs_pre",
        "abn_retpct_p5_vs_pre",
        "abn_retpct_p10_vs_pre",
        "abn_retpct_p20_vs_pre",
    ]
    if not all(c in panel.columns for c in needed):
        return

    xs = np.array([0, 1, 6, 11, 21], dtype=float)
    ys = [0.0]
    ci = [0.0]
    for c in needed:
        m, h = _mean_and_ci(panel[c])
        ys.append(m)
        ci.append(h if np.isfinite(h) else np.nan)

    ys = np.array(ys, dtype=float)
    ci = np.array(ci, dtype=float)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys, marker="o")
    if np.isfinite(ci[1:]).any():
        ax.fill_between(xs, ys - np.nan_to_num(ci), ys + np.nan_to_num(ci), alpha=0.15)

    ax.axhline(0, linewidth=1, alpha=0.4)
    ax.set_title("Average cumulative abnormal return (market-adjusted) around earnings")
    ax.set_xlabel("Trading days from PRE close (0=PRE, 1=REACT, 6=+5, 11=+10, 21=+20)")
    ax.set_ylabel("Abnormal return (%)")
    ax.grid(True, alpha=0.25)
    _savefig(fig, out_path, dpi)


def plot_mean_abnormal_drift_bars(
    panel: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    needed = [
        "abn_retpct_drift_5_vs_react",
        "abn_retpct_drift_10_vs_react",
        "abn_retpct_drift_20_vs_react",
    ]
    if not all(c in panel.columns for c in needed):
        return

    labels = ["5d", "10d", "20d"]
    means, cis = [], []
    for c in needed:
        m, h = _mean_and_ci(panel[c])
        means.append(m)
        cis.append(h if np.isfinite(h) else 0.0)

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labels, means, yerr=cis, capsize=4)
    ax.axhline(0, linewidth=1, alpha=0.4)
    ax.set_title("Average abnormal drift AFTER reaction date")
    ax.set_xlabel("Horizon from REACT")
    ax.set_ylabel("Abnormal drift return (%)")
    ax.grid(True, axis="y", alpha=0.25)
    _savefig(fig, out_path, dpi)


def plot_drift_histogram(
    panel: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    c = "abn_retpct_drift_20_vs_react"
    if c not in panel.columns:
        return
    v = pd.to_numeric(panel[c], errors="coerce").dropna()
    if v.empty:
        return

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(v, bins=12)
    ax.axvline(0, linewidth=1, alpha=0.4)
    ax.set_title("Distribution of 20-day abnormal drift (REACT→+20)")
    ax.set_xlabel("Abnormal drift return (%)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    _savefig(fig, out_path, dpi)


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple PEAD visuals: price+earnings markers, event-time abnormal curve, drift bars.")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--panel-name", default="event_panel")
    ap.add_argument("--out-subdir", default="viz")
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    panel = _read_panel(data_dir, ticker, args.panel_name)

    out_dir = data_dir / ticker / args.out_subdir
    ensure_dir(out_dir)

    prices = _read_prices(data_dir, ticker)
    if prices is not None:
        plot_price_with_earnings_markers(
            prices, panel,
            out_dir / f"{ticker}_01_price_with_earnings.png",
            dpi=args.dpi
        )

    plot_mean_abnormal_event_curve(
        panel,
        out_dir / f"{ticker}_02_mean_abn_curve_from_pre.png",
        dpi=args.dpi
    )
    plot_mean_abnormal_drift_bars(
        panel,
        out_dir / f"{ticker}_03_mean_abn_drift_bars.png",
        dpi=args.dpi
    )
    plot_drift_histogram(
        panel,
        out_dir / f"{ticker}_04_hist_abn_drift_20.png",
        dpi=args.dpi
    )

    # small run log
    (out_dir / f"{ticker}_31_viz_event_overview.meta.txt").write_text(
        f"ticker={ticker}\ncreated_at={now_iso()}\npanel={args.panel_name}\nrows={len(panel)}\n",
        encoding="utf-8"
    )

    print(f"[OK] {ticker}: wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
