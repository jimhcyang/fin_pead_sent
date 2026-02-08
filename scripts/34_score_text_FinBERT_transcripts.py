#!/usr/bin/env python3
# scripts/34_score_text_FinBERT_transcripts.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from _common import DEFAULT_20
from _finbert import load_finbert, score_finbert_batch, chunked


def read_tickers_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def clean_tickers(tickers: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in tickers:
        tt = t.strip().upper()
        if tt and tt not in seen:
            out.append(tt)
            seen.add(tt)
    return out


def _maybe_write(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    df.to_csv(path, index=False)


def agg_mean(df_units: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df_units.empty:
        return df_units
    cols = ["finbert_pos", "finbert_neg", "finbert_neu", "finbert_tone"]
    g = df_units.groupby(keys, dropna=False)[cols].mean().reset_index()
    return g


def pivot_wide(df_long: pd.DataFrame, suffix_prefix: str) -> pd.DataFrame:
    id_cols = ["ticker", "earnings_date", "day0_date"]
    val_cols = ["finbert_pos", "finbert_neg", "finbert_neu", "finbert_tone"]
    if df_long.empty:
        return pd.DataFrame(columns=id_cols)
    p = df_long.pivot_table(index=id_cols, columns=["section", "speaker_type"], values=val_cols, aggfunc="first")
    p.columns = [f"{suffix_prefix}{v}__{sec}__{typ}" for (v, sec, typ) in p.columns]
    p = p.reset_index()
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Score transcript text units with FinBERT.")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--tickers-file", default=None)
    ap.add_argument("--max-tickers", type=int, default=None)

    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--min-chars", type=int, default=10, help="Skip units shorter than this many characters.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-operator", action="store_true", help="Include operator turns (default drops them).")
    ap.add_argument("--model", default=None, help="HF model name/path (default env FINBERT_MODEL_NAME or ProsusAI/finbert)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    # tickers
    tickers: list[str] = []
    if args.ticker:
        tickers = [args.ticker]
    else:
        if args.tickers_file:
            tickers += read_tickers_file(Path(args.tickers_file))
        if args.tickers:
            tickers += args.tickers
        if not tickers:
            tickers = DEFAULT_20
    tickers = clean_tickers(tickers)
    if args.max_tickers is not None:
        tickers = tickers[: int(args.max_tickers)]

    tok, mdl, dev = load_finbert(model_name=args.model)
    bs = int(args.batch_size)

    for t in tickers:
        t = t.upper()
        in_path = data_dir / t / "events" / "text_units_transcripts.csv"
        if not in_path.exists():
            print(f"[WARN] {t}: missing {in_path} (run scripts/31 first)")
            continue

        df = pd.read_csv(in_path)
        if df.empty:
            print(f"[WARN] {t}: empty transcripts units")
            continue

        if (not args.keep_operator) and ("speaker_type" in df.columns):
            df = df[df["speaker_type"].astype(str).str.lower().ne("operator")].copy()

        texts: List[str] = df["text"].astype(str).tolist()
        mask_keep = [len(tx) >= int(args.min_chars) for tx in texts]
        df = df.loc[mask_keep].reset_index(drop=True)
        texts = df["text"].astype(str).tolist()

        scores_rows = []
        for chunk in chunked(texts, bs):
            scores_rows.extend(score_finbert_batch(chunk, tok=tok, mdl=mdl, device=dev))

        if len(scores_rows) != len(df):
            print(f"[ERR] {t}: scoring count mismatch ({len(scores_rows)} vs {len(df)})")
            continue

        out_units = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores_rows)], axis=1)
        out_dir = data_dir / t / "events"
        units_path = out_dir / "text_finbert_transcripts_units.csv"
        _maybe_write(out_units, units_path, overwrite=bool(args.overwrite))

        # Aggregations
        if not out_units.empty:
            if "section" not in out_units.columns:
                out_units["section"] = "all"
            if "speaker_type" not in out_units.columns:
                out_units["speaker_type"] = "all"

            d1 = out_units.copy()
            d2 = out_units.copy(); d2["speaker_type"] = "all"
            d3 = out_units.copy(); d3["section"] = "all"
            d4 = out_units.copy(); d4["section"] = "all"; d4["speaker_type"] = "all"

            ev_long = pd.concat([d1, d2, d3, d4], ignore_index=True)
            ev_long = agg_mean(ev_long, keys=["ticker", "earnings_date", "day0_date", "section", "speaker_type"])
        else:
            ev_long = out_units

        ev_path = out_dir / "text_finbert_transcripts_event_long.csv"
        _maybe_write(ev_long, ev_path, overwrite=bool(args.overwrite))
        ev_wide = pivot_wide(ev_long, suffix_prefix="tr_finbert_")
        ev_wide_path = out_dir / "text_finbert_transcripts_event_wide.csv"
        _maybe_write(ev_wide, ev_wide_path, overwrite=bool(args.overwrite))

        meta = {
            "ticker": t,
            "model": args.model or "ProsusAI/finbert",
            "in_path": str(in_path),
            "units_out": str(units_path),
            "event_long_out": str(ev_path),
            "event_wide_out": str(ev_wide_path),
            "min_chars": int(args.min_chars),
            "keep_operator": bool(args.keep_operator),
            "rows_in": int(len(df)),
            "rows_scored": int(len(out_units)),
            "events_scored": int(ev_wide.shape[0]) if not ev_wide.empty else 0,
        }
        (out_dir / "text_finbert_transcripts.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"[OK] {t}: FinBERT transcript units -> {units_path} rows={len(out_units):,}")
        print(f"[OK] {t}: FinBERT transcript event -> {ev_wide_path} events={meta['events_scored']}")


if __name__ == "__main__":
    main()
