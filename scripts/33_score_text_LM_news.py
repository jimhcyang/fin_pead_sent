#!/usr/bin/env python3
# scripts/33_score_text_LM_news.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _common import DEFAULT_20
from _lm import load_lm_sets, score_lm


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


def agg_event_phase(df_units: pd.DataFrame) -> pd.DataFrame:
    cols_sum = [
        "n_tokens", "n_tokens_raw",
        "lm_pos", "lm_neg", "lm_unc", "lm_lit", "lm_constr", "lm_strong", "lm_weak", "lm_complex"
    ]
    keys = ["ticker", "earnings_date", "day0_date", "phase"]

    g = df_units.groupby(keys, dropna=False)[cols_sum].sum().reset_index()
    n = g["n_tokens"].astype(float).replace(0.0, pd.NA)

    g["lm_pos_prop"] = g["lm_pos"] / n
    g["lm_neg_prop"] = g["lm_neg"] / n
    g["lm_unc_prop"] = g["lm_unc"] / n
    g["lm_lit_prop"] = g["lm_lit"] / n
    g["lm_constr_prop"] = g["lm_constr"] / n
    g["lm_tone"] = (g["lm_pos"] - g["lm_neg"]) / n
    return g


def pivot_wide(df_long: pd.DataFrame, suffix_prefix: str) -> pd.DataFrame:
    id_cols = ["ticker", "earnings_date", "day0_date"]
    val_cols = ["n_tokens", "lm_pos_prop", "lm_neg_prop", "lm_unc_prop", "lm_lit_prop", "lm_constr_prop", "lm_tone"]
    if df_long.empty:
        return pd.DataFrame(columns=id_cols)

    p = df_long.pivot_table(index=id_cols, columns="phase", values=val_cols, aggfunc="first")
    p.columns = [f"{suffix_prefix}{v}__{ph}" for (v, ph) in p.columns]
    p = p.reset_index()

    # add deltas (post - pre) for key fields if both exist
    for v in ["lm_pos_prop", "lm_neg_prop", "lm_unc_prop", "lm_lit_prop", "lm_constr_prop", "lm_tone"]:
        a = f"{suffix_prefix}{v}__post"
        b = f"{suffix_prefix}{v}__pre"
        if a in p.columns and b in p.columns:
            p[f"{suffix_prefix}{v}__post_minus_pre"] = p[a] - p[b]
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Score news text units with LM dictionary.")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--tickers-file", default=None)
    ap.add_argument("--max-tickers", type=int, default=None)

    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--min-tokens", type=int, default=20, help="Minimum LM-valid tokens to keep a unit.")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()
    data_dir = Path(args.data_dir)

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

    lm_sets = load_lm_sets(data_dir=str(data_dir))

    for t in tickers:
        t = t.upper()
        in_path = data_dir / t / "events" / "text_units_news.csv"
        if not in_path.exists():
            print(f"[WARN] {t}: missing {in_path} (run scripts/31 first)")
            continue

        df = pd.read_csv(in_path)
        if df.empty:
            print(f"[WARN] {t}: empty news units")
            continue

        scored_rows = []
        for _, r in df.iterrows():
            text = str(r.get("text", "") or "")
            s = score_lm(text, lm_sets)
            if int(s["n_tokens"]) < int(args.min_tokens):
                continue
            out = dict(r)
            out.update(s)
            scored_rows.append(out)

        out_units = pd.DataFrame(scored_rows)
        out_dir = data_dir / t / "events"
        units_path = out_dir / "text_lm_news_units.csv"
        _maybe_write(out_units, units_path, overwrite=bool(args.overwrite))

        # aggregate pre/post
        ev_ph = agg_event_phase(out_units)

        # add "all" phase aggregation
        if not out_units.empty:
            tmp = out_units.copy()
            tmp["phase"] = "all"
            ev_all = agg_event_phase(tmp)
            ev_long = pd.concat([ev_ph, ev_all], ignore_index=True)
        else:
            ev_long = ev_ph

        ev_path = out_dir / "text_lm_news_event_long.csv"
        _maybe_write(ev_long, ev_path, overwrite=bool(args.overwrite))

        # legacy filename for backward compatibility
        _maybe_write(ev_long, out_dir / "text_lm_news_event.csv", overwrite=bool(args.overwrite))

        ev_wide = pivot_wide(ev_long, suffix_prefix="nw_")
        ev_wide_path = out_dir / "text_lm_news_event_wide.csv"
        _maybe_write(ev_wide, ev_wide_path, overwrite=bool(args.overwrite))

        meta = {
            "ticker": t,
            "in_path": str(in_path),
            "units_out": str(units_path),
            "event_long_out": str(ev_path),
            "event_wide_out": str(ev_wide_path),
            "min_tokens": int(args.min_tokens),
            "rows_in": int(len(df)),
            "rows_scored": int(len(out_units)),
            "events_scored": int(ev_wide.shape[0]) if not ev_wide.empty else 0,
        }
        (out_dir / "text_lm_news.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"[OK] {t}: wrote LM news units -> {units_path} rows={len(out_units):,}")
        print(f"[OK] {t}: wrote LM news event -> {ev_wide_path} events={meta['events_scored']}")


if __name__ == "__main__":
    main()
