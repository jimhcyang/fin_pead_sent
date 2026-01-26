#!/usr/bin/env python3
# scripts/35_viz_event_paths.py

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
        raise FileNotFoundError(f"Missing panel: {p} (run scripts/30_build_events_all.py first)")
    df = pd.read_csv(p)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    df["announce_timing"] = _coalesce_timing(df)
    return df


def _savefig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _row_to_path(row: pd.Series, cols: list[str]) -> np.ndarray:
    vals = [pd.to_numeric(row.get(c), errors="coerce") for c in cols]
    return np.array(vals, dtype=float)


def _nanmean_path(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    return np.nanmean(mat, axis=0)


def plot_spaghetti_abn_total_from_pre(df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    cols = [
        "abn_retpct_react_vs_pre",
        "abn_retpct_p5_vs_pre",
        "abn_retpct_p10_vs_pre",
        "abn_retpct_p20_vs_pre",
    ]
    if not all(c in df.columns for c in cols):
        return

    xs = np.array([0, 1, 6, 11, 21], dtype=float)  # PRE, REACT, +5, +10, +20 (in trading days)
    paths = []

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    for _, row in df.iterrows():
        y = np.concatenate(([0.0], _row_to_path(row, cols)))
        if np.isfinite(y).sum() < 3:
            continue
        paths.append(y)
        timing = str(row.get("announce_timing") or "").lower()
        marker = "x" if timing == "amc" else "o" if timing == "bmo" else "."
        ax.plot(xs, y, alpha=0.25, marker=marker, markersize=3)

    if not paths:
        return

    mean_y = _nanmean_path(np.vstack(paths))
    ax.plot(xs, mean_y, linewidth=3, alpha=0.9)

    ax.axhline(0, linewidth=1, alpha=0.35)
    ax.set_title("Event-time abnormal return paths (PRE → horizons) — each line is one earnings event")
    ax.set_xlabel("Trading days from PRE close (0=PRE, 1=REACT, 6=+5, 11=+10, 21=+20)")
    ax.set_ylabel("Cumulative abnormal return (%)")
    ax.grid(True, alpha=0.25)
    _savefig(fig, out_path, dpi)


def plot_spaghetti_abn_drift_from_react(df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    cols = [
        "abn_retpct_drift_5_vs_react",
        "abn_retpct_drift_10_vs_react",
        "abn_retpct_drift_20_vs_react",
    ]
    if not all(c in df.columns for c in cols):
        return

    xs = np.array([0, 5, 10, 20], dtype=float)  # REACT, +5, +10, +20
    paths = []

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    for _, row in df.iterrows():
        y = np.concatenate(([0.0], _row_to_path(row, cols)))
        if np.isfinite(y).sum() < 3:
            continue
        paths.append(y)
        timing = str(row.get("announce_timing") or "").lower()
        marker = "x" if timing == "amc" else "o" if timing == "bmo" else "."
        ax.plot(xs, y, alpha=0.25, marker=marker, markersize=3)

    if not paths:
        return

    mean_y = _nanmean_path(np.vstack(paths))
    ax.plot(xs, mean_y, linewidth=3, alpha=0.9)

    ax.axhline(0, linewidth=1, alpha=0.35)
    ax.set_title("Event-time abnormal DRIFT paths (REACT → horizons) — each line is one earnings event")
    ax.set_xlabel("Trading days from REACT close (0=REACT, 5=+5, 10=+10, 20=+20)")
    ax.set_ylabel("Abnormal drift return (%)")
    ax.grid(True, alpha=0.25)
    _savefig(fig, out_path, dpi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Event-ordered ‘spaghetti’ plots: each earnings event as its own abnormal-return path."
    )
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--panel-name", default="event_panel")
    ap.add_argument("--out-subdir", default="viz")
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    df = _read_panel(data_dir, ticker, args.panel_name).sort_values("earnings_date")

    out_dir = data_dir / ticker / args.out_subdir
    ensure_dir(out_dir)

    plot_spaghetti_abn_total_from_pre(
        df,
        out_dir / f"{ticker}_10_spaghetti_abn_total_paths.png",
        dpi=args.dpi,
    )
    plot_spaghetti_abn_drift_from_react(
        df,
        out_dir / f"{ticker}_11_spaghetti_abn_drift_paths.png",
        dpi=args.dpi,
    )

    (out_dir / f"{ticker}_35_viz_event_paths.meta.txt").write_text(
        f"ticker={ticker}\ncreated_at={now_iso()}\npanel={args.panel_name}\nrows={len(df)}\n",
        encoding="utf-8",
    )
    print(f"[OK] {ticker}: wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
