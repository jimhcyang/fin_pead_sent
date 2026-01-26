#!/usr/bin/env python3
# scripts/36_viz_event_cards.py

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
    df["ticker"] = ticker
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    df["announce_timing"] = _coalesce_timing(df)
    return df


def _savefig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _fmt_pct(x) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.2f}%"
    except Exception:
        return "NA"


def _fmt_date(x) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return "NA"


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-event cards: price path + return heatmap (stock/market/abnormal).")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--panel-name", default="event_panel")
    ap.add_argument("--out-subdir", default="viz")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    df = _read_panel(data_dir, ticker, args.panel_name).sort_values("earnings_date")

    out_base = data_dir / ticker / args.out_subdir / "events"
    ensure_dir(out_base)

    # Columns we want (robust if some are missing)
    price_cols = [
        ("PRE", "pre_adj_close"),
        ("REACT", "react_adj_close"),
        ("+5", "adj_close_p5"),
        ("+10", "adj_close_p10"),
        ("+20", "adj_close_p20"),
    ]

    cols_stock = [
        ("PRE→REACT", "retpct_react_vs_pre"),
        ("PRE→+5", "retpct_p5_vs_pre"),
        ("PRE→+10", "retpct_p10_vs_pre"),
        ("PRE→+20", "retpct_p20_vs_pre"),
        ("REACT→+5", "retpct_drift_5_vs_react"),
        ("REACT→+10", "retpct_drift_10_vs_react"),
        ("REACT→+20", "retpct_drift_20_vs_react"),
    ]
    cols_mkt = [
        ("PRE→REACT", "mkt_retpct_react_vs_pre"),
        ("PRE→+5", "mkt_retpct_p5_vs_pre"),
        ("PRE→+10", "mkt_retpct_p10_vs_pre"),
        ("PRE→+20", "mkt_retpct_p20_vs_pre"),
        ("REACT→+5", "mkt_retpct_drift_5_vs_react"),
        ("REACT→+10", "mkt_retpct_drift_10_vs_react"),
        ("REACT→+20", "mkt_retpct_drift_20_vs_react"),
    ]
    cols_abn = [
        ("PRE→REACT", "abn_retpct_react_vs_pre"),
        ("PRE→+5", "abn_retpct_p5_vs_pre"),
        ("PRE→+10", "abn_retpct_p10_vs_pre"),
        ("PRE→+20", "abn_retpct_p20_vs_pre"),
        ("REACT→+5", "abn_retpct_drift_5_vs_react"),
        ("REACT→+10", "abn_retpct_drift_10_vs_react"),
        ("REACT→+20", "abn_retpct_drift_20_vs_react"),
    ]

    has_mkt = all(c in df.columns for _, c in cols_mkt)
    has_abn = all(c in df.columns for _, c in cols_abn)

    n_written = 0
    for _, row in df.iterrows():
        ed = row["earnings_date"]
        if pd.isna(ed):
            continue
        eds = ed.strftime("%Y-%m-%d")
        out_path = out_base / f"{eds}_event_card.png"
        if out_path.exists() and not args.overwrite:
            continue

        # --- Price path ---
        px_labels, px_vals = [], []
        for lab, c in price_cols:
            if c in df.columns:
                px_labels.append(lab)
                px_vals.append(pd.to_numeric(row.get(c), errors="coerce"))
        px_vals = np.array(px_vals, dtype=float)

        # --- Return heatmap matrix ---
        xlabs = [lab for lab, _ in cols_stock]
        mat_rows = []
        ylabels = []

        # Stock row always if present
        stock_vals = [pd.to_numeric(row.get(c), errors="coerce") for _, c in cols_stock]
        mat_rows.append(stock_vals)
        ylabels.append("stock")

        if has_mkt:
            mkt_vals = [pd.to_numeric(row.get(c), errors="coerce") for _, c in cols_mkt]
            mat_rows.append(mkt_vals)
            ylabels.append("market")

        if has_abn:
            abn_vals = [pd.to_numeric(row.get(c), errors="coerce") for _, c in cols_abn]
            mat_rows.append(abn_vals)
            ylabels.append("abnormal")

        mat = np.array(mat_rows, dtype=float)

        # --- Header info ---
        timing = row.get("announce_timing")
        eps_s = row.get("eps_surprise_pct", np.nan)
        rev_s = row.get("revenue_surprise_pct", np.nan)

        pre_date = _fmt_date(row.get("pre_date"))
        react_date = _fmt_date(row.get("react_date"))
        d5 = _fmt_date(row.get("d5"))
        d10 = _fmt_date(row.get("d10"))
        d20 = _fmt_date(row.get("d20"))

        title = f"{ticker} earnings {eds} | timing={timing if pd.notna(timing) else 'NA'} | EPS surpr={_fmt_pct(eps_s)} | Rev surpr={_fmt_pct(rev_s)}"

        # --- Figure layout ---
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.1, 1.2])

        # Price path (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        if len(px_vals) >= 2 and np.isfinite(px_vals).any():
            ax1.plot(np.arange(len(px_vals)), px_vals, marker="o")
            ax1.set_xticks(np.arange(len(px_labels)))
            ax1.set_xticklabels(px_labels)
        ax1.set_title("Adj close at anchor points")
        ax1.set_xlabel("Anchor")
        ax1.set_ylabel("Adj close")
        ax1.grid(True, alpha=0.25)

        # Heatmap (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(mat, aspect="auto")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_xticks(np.arange(len(xlabs)))
        ax2.set_xticklabels(xlabs, rotation=30, ha="right")
        ax2.set_yticks(np.arange(len(ylabels)))
        ax2.set_yticklabels(ylabels)
        ax2.set_title("Returns (%) over event windows")

        # Text box (bottom full width)
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis("off")
        txt = (
            f"{title}\n\n"
            f"Anchors:\n"
            f"  PRE={pre_date}  REACT={react_date}  +5={d5}  +10={d10}  +20={d20}\n\n"
            f"Key abnormal drift (REACT→+20): {_fmt_pct(row.get('abn_retpct_drift_20_vs_react', np.nan))}\n"
            f"Key abnormal total (PRE→+20):  {_fmt_pct(row.get('abn_retpct_p20_vs_pre', np.nan))}\n"
        )
        ax3.text(0.01, 0.98, txt, va="top")

        fig.suptitle(title, y=1.02)
        _savefig(fig, out_path, dpi=args.dpi)
        n_written += 1

    (out_base / f"{ticker}_36_event_cards.meta.txt").write_text(
        f"ticker={ticker}\ncreated_at={now_iso()}\npanel={args.panel_name}\nrows={len(df)}\nwritten={n_written}\n",
        encoding="utf-8",
    )

    print(f"[OK] {ticker}: wrote {n_written} event cards -> {out_base}")


if __name__ == "__main__":
    main()
