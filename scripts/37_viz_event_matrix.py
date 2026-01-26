#!/usr/bin/env python3
# scripts/37_viz_event_matrix.py

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


def _plot_matrix(df: pd.DataFrame, cols: list[tuple[str, str]], title: str, out_path: Path, dpi: int) -> None:
    keep = [c for _, c in cols if c in df.columns]
    if len(keep) < 2:
        return

    mat = np.vstack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in keep]).T
    ylabels = df["earnings_date"].dt.strftime("%Y-%m-%d").fillna("NA").tolist()
    xlabels = [lab for lab, c in cols if c in df.columns]

    fig = plt.figure(figsize=(max(7, 1.3 * len(xlabels)), max(6, 0.25 * len(ylabels))))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(mat, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)

    _savefig(fig, out_path, dpi)


def main() -> None:
    ap = argparse.ArgumentParser(description="Event-ordered (by date) heatmaps for abnormal return windows and surprises.")
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

    total_cols = [
        ("REACT", "abn_retpct_react_vs_pre"),
        ("+5", "abn_retpct_p5_vs_pre"),
        ("+10", "abn_retpct_p10_vs_pre"),
        ("+20", "abn_retpct_p20_vs_pre"),
    ]
    drift_cols = [
        ("REACT→+5", "abn_retpct_drift_5_vs_react"),
        ("REACT→+10", "abn_retpct_drift_10_vs_react"),
        ("REACT→+20", "abn_retpct_drift_20_vs_react"),
    ]
    surprise_cols = [
        ("EPS surpr (%)", "eps_surprise_pct"),
        ("Rev surpr (%)", "revenue_surprise_pct"),
    ]

    _plot_matrix(
        df,
        total_cols,
        "Abnormal cumulative return by event (date order, PRE→h)",
        out_dir / f"{ticker}_12_matrix_abn_total_by_event.png",
        dpi=args.dpi,
    )
    _plot_matrix(
        df,
        drift_cols,
        "Abnormal drift by event (date order, REACT→h)",
        out_dir / f"{ticker}_13_matrix_abn_drift_by_event.png",
        dpi=args.dpi,
    )
    _plot_matrix(
        df,
        surprise_cols,
        "Surprises by event (date order)",
        out_dir / f"{ticker}_14_matrix_surprises_by_event.png",
        dpi=args.dpi,
    )

    (out_dir / f"{ticker}_37_viz_event_matrix.meta.txt").write_text(
        f"ticker={ticker}\ncreated_at={now_iso()}\npanel={args.panel_name}\nrows={len(df)}\n",
        encoding="utf-8",
    )
    print(f"[OK] {ticker}: wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
