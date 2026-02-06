#!/usr/bin/env python3
# scripts/40_run_text_pipeline_all.py

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed (exit={r.returncode}): {' '.join(cmd)}")


def _agg_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").dropna().mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LM text pipeline (extract -> score -> aggregate).")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--pre-bdays", type=int, default=5)
    ap.add_argument("--post-bdays", type=int, default=10)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--out", default="data/_derived/text/event_text_features_lm.csv")
    args = ap.parse_args()

    py = sys.executable
    scripts_dir = Path(__file__).resolve().parent

    # 1) extract
    if not args.skip_extract:
        cmd = [py, str(scripts_dir / "31_extract_text_units.py"), "--data-dir", args.data_dir,
               "--pre-bdays", str(int(args.pre_bdays)), "--post-bdays", str(int(args.post_bdays))]
        if args.ticker:
            cmd += ["--ticker", args.ticker]
        elif args.tickers:
            cmd += ["--tickers", *args.tickers]
        run(cmd)

    # 2) score
    if not args.skip_score:
        run([py, str(scripts_dir / "32_score_text_LM_transcript.py"), "--data-dir", args.data_dir])
        run([py, str(scripts_dir / "33_score_text_LM_news.py"), "--data-dir", args.data_dir, "--only-window"])

    # 3) aggregate (event-level)
    news_path = Path("data/_derived/text/lm_units_news.csv")
    tr_path = Path("data/_derived/text/lm_units_transcript.csv")
    if not news_path.exists() and not tr_path.exists():
        raise FileNotFoundError("No LM unit outputs found; run extraction/scoring first.")

    dfN = pd.read_csv(news_path) if news_path.exists() else pd.DataFrame()
    dfT = pd.read_csv(tr_path) if tr_path.exists() else pd.DataFrame()

    # filter tickers if requested
    want = None
    if args.ticker:
        want = {args.ticker.upper()}
    elif args.tickers:
        want = {t.upper() for t in args.tickers}
    if want:
        if not dfN.empty:
            dfN = dfN[dfN["ticker"].isin(want)].copy()
        if not dfT.empty:
            dfT = dfT[dfT["ticker"].isin(want)].copy()

    # build event universe
    events = set()
    if not dfN.empty:
        events |= set(zip(dfN["ticker"], dfN["earnings_date"]))
    if not dfT.empty:
        events |= set(zip(dfT["ticker"], dfT["earnings_date"]))

    rows: List[Dict] = []

    for tkr, ed in sorted(events):
        out = {"ticker": tkr, "earnings_date": ed}

        # NEWS: pre/event/post
        if not dfN.empty:
            g = dfN[(dfN["ticker"] == tkr) & (dfN["earnings_date"] == ed)].copy()
            if not g.empty:
                pre = g[g["bd_offset"] < 0]
                post = g[g["bd_offset"] > 0]
                day0 = g[g["bd_offset"] == 0]

                out["news_article_count_pre"] = int(len(pre))
                out["news_article_count_post"] = int(len(post))
                out["news_article_count_day0"] = int(len(day0))

                for metric in ["lm_tone", "lm_neg_prop", "lm_pos_prop", "lm_unc_prop", "lm_lit_prop"]:
                    out[f"news_{metric}_pre_mean"] = _agg_mean(pre, metric)
                    out[f"news_{metric}_post_mean"] = _agg_mean(post, metric)
                    out[f"news_{metric}_day0_mean"] = _agg_mean(day0, metric)

                    a = out[f"news_{metric}_post_mean"]
                    b = out[f"news_{metric}_pre_mean"]
                    out[f"news_{metric}_post_minus_pre"] = (a - b) if (np.isfinite(a) and np.isfinite(b)) else np.nan

        # TRANSCRIPT: prepared vs qa (or full)
        if not dfT.empty:
            g = dfT[(dfT["ticker"] == tkr) & (dfT["earnings_date"] == ed)].copy()
            if not g.empty:
                prep = g[g["unit_label"] == "prepared"]
                qa = g[g["unit_label"] == "qa"]
                full = g[g["unit_label"] == "full"]

                # prefer prepared/qa if present; else full
                if not prep.empty:
                    out["tr_lm_tone_prepared"] = _agg_mean(prep, "lm_tone")
                    out["tr_lm_neg_prop_prepared"] = _agg_mean(prep, "lm_neg_prop")
                if not qa.empty:
                    out["tr_lm_tone_qa"] = _agg_mean(qa, "lm_tone")
                    out["tr_lm_neg_prop_qa"] = _agg_mean(qa, "lm_neg_prop")
                if prep.empty and qa.empty and not full.empty:
                    out["tr_lm_tone_full"] = _agg_mean(full, "lm_tone")
                    out["tr_lm_neg_prop_full"] = _agg_mean(full, "lm_neg_prop")

                if ("tr_lm_tone_qa" in out) and ("tr_lm_tone_prepared" in out):
                    a = out.get("tr_lm_tone_qa")
                    b = out.get("tr_lm_tone_prepared")
                    out["tr_lm_tone_qa_minus_prepared"] = (a - b) if (np.isfinite(a) and np.isfinite(b)) else np.nan

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # quick diagnostic note
    n_events = len(out_df)
    n_news_missing = int(out_df["news_article_count_pre"].isna().sum()) if "news_article_count_pre" in out_df.columns else n_events
    n_tr_missing = int(out_df.filter(like="tr_lm_").isna().all(axis=1).sum()) if any(c.startswith("tr_lm_") for c in out_df.columns) else n_events

    print(f"[OK] wrote event-level LM text features -> {out_path} rows={n_events:,}")
    print(f"[NOTE] events_missing_news={n_news_missing:,} events_missing_transcript={n_tr_missing:,}")


if __name__ == "__main__":
    main()
