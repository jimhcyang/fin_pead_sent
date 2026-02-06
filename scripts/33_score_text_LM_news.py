#!/usr/bin/env python3
# scripts/33_score_text_LM_news.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _lm import load_lm_sets, score_lm


def main() -> None:
    ap = argparse.ArgumentParser(description="Score news article text units using LM dictionary.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--in", dest="inp", default="data/_derived/text/text_units_news.csv")
    ap.add_argument("--out", default="data/_derived/text/lm_units_news.csv")
    ap.add_argument("--min-tokens", type=int, default=30)
    ap.add_argument("--only-window", action="store_true", help="Keep only units within [-5,+10] business-day window.")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(f"Missing input units: {inp} (run scripts/31_extract_text_units.py first)")

    df = pd.read_csv(inp)
    if df.empty:
        raise RuntimeError("No news units found.")

    if args.only_window and "in_m5_p10" in df.columns:
        df = df[df["in_m5_p10"] == True].copy()

    lm_sets = load_lm_sets(data_dir=args.data_dir)

    rows = []
    for _, r in df.iterrows():
        text = str(r.get("text") or "").strip()
        sc = score_lm(text, lm_sets)
        if int(sc["n_tokens"]) < int(args.min_tokens):
            continue
        out = dict(r)
        out.update(sc)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[OK] wrote LM-scored news units -> {out_path} rows={len(out_df):,}")


if __name__ == "__main__":
    main()
