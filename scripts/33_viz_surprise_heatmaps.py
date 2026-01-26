#!/usr/bin/env python3
# scripts/33_viz_surprise_heatmaps.py

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


def _heatmap(mat: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, out_path: Path, dpi: int, ylabel: str) -> None:
    fig = plt.figure(figsize=(8, max(4.5, 0.35 * len(row_labels))))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(mat, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")

    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel(ylabel)
    _savefig(fig, out_path, dpi)


def main() -> None:
    ap = argparse.ArgumentParser(description="Heatmaps: surprise deciles x horizons for abnormal returns (robust to missing horizons).")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--panel-name", default="event_panel")
    ap.add_argument("--out-subdir", default="viz")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--deciles", type=int, default=10)
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    df = _read_panel(data_dir, ticker, args.panel_name).copy()

    out_dir = data_dir / ticker / args.out_subdir
    ensure_dir(out_dir)

    if "eps_surprise_pct" not in df.columns:
        print(f"[WARN] {ticker}: eps_surprise_pct not found; skipping.")
        return

    eps = pd.to_numeric(df["eps_surprise_pct"], errors="coerce")
    df["eps_decile"] = pd.qcut(eps, q=args.deciles, labels=False, duplicates="drop")
    if df["eps_decile"].isna().all():
        print(f"[WARN] {ticker}: not enough variation for deciles; skipping.")
        return
    df["eps_decile"] = df["eps_decile"].astype("Int64") + 1  # 1..K
    K = int(df["eps_decile"].max())
    if K < 2:
        print(f"[WARN] {ticker}: deciles collapsed to K={K}; skipping.")
        return

    # Canonical horizon sets (we will use the subset that exists)
    drift_cols_all = [
        ("5d", "abn_retpct_drift_5_vs_react"),
        ("10d", "abn_retpct_drift_10_vs_react"),
        ("20d", "abn_retpct_drift_20_vs_react"),
    ]
    total_cols_all = [
        ("REACT", "abn_retpct_react_vs_pre"),
        ("+5", "abn_retpct_p5_vs_pre"),
        ("+10", "abn_retpct_p10_vs_pre"),
        ("+20", "abn_retpct_p20_vs_pre"),
    ]

    def build_mat(cols_all):
        # keep only columns that exist
        cols = [(lab, c) for (lab, c) in cols_all if c in df.columns]
        if len(cols) == 0:
            return None, None
        row_labels = [str(i) for i in range(1, K + 1)]
        col_labels = [lab for lab, _ in cols]
        mat = np.full((K, len(cols)), np.nan, dtype=float)
        for i in range(1, K + 1):
            sub = df[df["eps_decile"] == i]
            for j, (_, c) in enumerate(cols):
                mat[i - 1, j] = float(pd.to_numeric(sub[c], errors="coerce").mean())
        return mat, col_labels

    mat_drift, labels_drift = build_mat(drift_cols_all)
    if mat_drift is not None:
        _heatmap(
            mat_drift,
            [str(i) for i in range(1, K + 1)],
            labels_drift,
            "Abnormal DRIFT returns by EPS-surprise decile",
            out_dir / f"{ticker}_07_heatmap_eps_decile_x_abn_drift.png",
            dpi=args.dpi,
            ylabel="EPS surprise decile (1=most negative)",
        )

    mat_total, labels_total = build_mat(total_cols_all)
    if mat_total is not None:
        _heatmap(
            mat_total,
            [str(i) for i in range(1, K + 1)],
            labels_total,
            "Cumulative abnormal returns by EPS-surprise decile",
            out_dir / f"{ticker}_08_heatmap_eps_decile_x_abn_total.png",
            dpi=args.dpi,
            ylabel="EPS surprise decile (1=most negative)",
        )

    (out_dir / f"{ticker}_33_viz_surprise_heatmaps.meta.txt").write_text(
        f"ticker={ticker}\ncreated_at={now_iso()}\npanel={args.panel_name}\nrows={len(df)}\ndeciles={args.deciles}\nK={K}\n",
        encoding="utf-8",
    )
    print(f"[OK] {ticker}: wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
