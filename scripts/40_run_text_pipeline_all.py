#!/usr/bin/env python3
from __future__ import annotations

"""
40_run_text_pipeline_all.py

Orchestrates the "text → LM features → merged event table" pipeline:

1) scripts/31_extract_text_units.py
   - Builds unit-level tables for news + transcripts per event.

2) scripts/32_score_text_LM_transcripts.py
   - Scores transcript units with a LM and aggregates to event-level (wide + long).

3) scripts/33_score_text_LM_news.py
   - Scores news units with a LM and aggregates to event-level (wide + long).

4) Merges event-level text features into a single panel:
   data/_derived/text/event_text_features.csv

Optionally merges per-event return/abnormal-return tables (produced by other scripts),
so we can compute correlations between text features and return outcomes.

Outputs (in repo root, unless --out-dir is set):
- data/_derived/text/event_text_features.csv   (long panel; ticker × events)
- summary_by_ticker.csv                       (means by ticker)
- summary_by_event_seq.csv                    (means by event sequence index 1..N across tickers)
- top_correlations.csv                        (top correlations for the chosen corr target)
- top_correlations_by_target.csv              (top correlations for every detected return target)

Notes
- Return tables are expected under: data/<TICKER>/events/<RETURN_FILE>.csv
- Join keys: ticker, earnings_date, day0_date (optionally day0_dt)
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd

try:
    from _common import DEFAULT_20
except Exception:
    DEFAULT_20 = [
        "NVDA","GOOGL","AAPL","MSFT","AMZN","META","AVGO","TSLA","LLY","WMT",
        "JPM","V","XOM","JNJ","ORCL","MA","MU","COST","AMD","ABBV",
    ]


# --------------------------
# Utilities
# --------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        raise RuntimeError(f"Command failed (exit={r.returncode}): {' '.join(cmd)}")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_csv_maybe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed reading {path}: {e}", flush=True)
        return None
    if df.empty:
        return None
    return df


def _ensure_str_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def _merge_on_keys(left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str], prefix_on_overlap: str) -> pd.DataFrame:
    """
    Merge right into left on keys.
    If columns overlap (excluding keys), prefix right's overlapping columns.
    """
    overlap = [c for c in right.columns if c in left.columns and c not in keys]
    if overlap:
        right = right.rename(columns={c: f"{prefix_on_overlap}__{c}" for c in overlap})
    return left.merge(right, on=list(keys), how="left")


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            cols.append(c)
            continue
        # try coercion
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().sum() >= max(5, int(0.6 * len(df))):
            df[c] = coerced
            cols.append(c)
    return cols


def _is_text_feature_col(c: str) -> bool:
    """Heuristic: which columns are 'text features' worth correlating with returns.

    Your LM scoring scripts emit many columns like:
      tr__tr_lm_neg_prop__qa__all, news__nw_lm_tone__post, ...

    The previous suffix-based filter (endswith '_mean', '_prop', ...) missed these
    because they often end with '__all'. We instead key off prefixes + the presence
    of 'lm_' / tone / sent tokens in the name.
    """
    cl = c.lower()

    # LM features
    if cl.startswith("lm_"):
        return True
    if cl.startswith(("tr__", "news__")) and ("lm_" in cl or "tone" in cl or "sent" in cl):
        return True
    if cl.startswith(("trfb__", "newsfb__", "troa__", "newsoa__")):
        return True

    # counts emitted by scoring scripts
    if cl in ("n_units", "n_tokens", "n_units_pre", "n_units_post", "n_units_prepared", "n_units_qa"):
        return True

    # engineered deltas/ratios (if present)
    if cl.startswith(("delta_", "ratio_")):
        return True

    return False


def _detect_return_targets(df: pd.DataFrame) -> List[str]:
    """
    Identify columns that look like return / abnormal-return outcomes.
    """
    pats = [
        r"^ar_",
        r"^car_",
        r"^tret_",
        r"^spxret_",
        r"^cret_",
        r"^abret",
        r"^return",
        r"^ret_",
    ]
    out = []
    for c in df.columns:
        cl = c.lower()
        if any(re.match(p, cl) for p in pats):
            # exclude metadata-ish ret columns
            if cl in ("ret_n", "return_n"):
                continue
            out.append(c)
    return out


def _pick_corr_target(df: pd.DataFrame, user_target: Optional[str]) -> Optional[str]:
    # If user specifies a target, try exact match first, then resolve common aliases.
    if user_target:
        if user_target in df.columns:
            return user_target

        ut = str(user_target).strip()
        ut_l = ut.lower()

        # Common alias: car_0_10 -> car_*_p0_p10_*
        m = re.match(r"^(car|ar|tret|cret)_(\-?\d+)_([0-9]+)$", ut_l)
        if m:
            kind, a_s, b_s = m.group(1), m.group(2), m.group(3)
            a, b = int(a_s), int(b_s)

            def _p(x: int) -> str:
                return f"p{x}" if x >= 0 else f"m{abs(x)}"

            a_tag, b_tag = _p(a), _p(b)
            subs = [
                f"{a_tag}_{b_tag}",
                f"{a_tag}-{b_tag}",
                f"{a_tag}to{b_tag}",
                f"{a_tag}{b_tag}",
            ]

            candidates = [
                c for c in df.columns
                if c.lower().startswith(kind) and any(s in c.lower() for s in subs)
            ]
            if candidates:
                # Prefer market-model CAR over simple CAR when both exist.
                def _rank(c: str) -> tuple[int, int, int]:
                    lc = c.lower()
                    return (
                        0 if "mm" in lc else 1,
                        0 if "simple" in lc else 1,
                        len(c),
                    )
                return sorted(candidates, key=_rank)[0]

        # Last-resort: case-insensitive match
        col_map = {c.lower(): c for c in df.columns}
        return col_map.get(ut_l)

    preferred = [
        # abnormal returns (best for PEAD/event studies)
        "car_0_5", "car_0_10", "car_0_20",
        "ar_0_1", "ar_0_5", "ar_0_10", "ar_0_20",
        # benchmark-adjusted cumulative return (if present)
        "cret_0_5", "cret_0_10", "cret_0_20",
        # raw total return (fallback)
        "tret_0_5", "tret_0_10", "tret_0_20",
    ]
    for c in preferred:
        if c in df.columns:
            return c
    # fallback to first detected target
    targets = _detect_return_targets(df)
    return targets[0] if targets else None

def _top_correlations(
    df: pd.DataFrame,
    y_col: str,
    candidate_cols: Sequence[str],
    min_n: int = 80,
) -> pd.DataFrame:
    """
    Compute correlations corr(feature, y_col) for candidate feature columns.
    Returns a sorted dataframe with columns: feature, corr, n.
    """
    rows = []
    y = pd.to_numeric(df[y_col], errors="coerce")
    for c in candidate_cols:
        if c == y_col:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n < min_n:
            continue
        corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
        if np.isfinite(corr):
            rows.append({"feature": c, "corr": corr, "n": n})
    if not rows:
        return pd.DataFrame(columns=["feature", "corr", "n"])
    out = pd.DataFrame(rows).sort_values("corr", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return out


# --------------------------
# Main
# --------------------------

@dataclass
class TickerMeta:
    ticker: str
    events: int
    has_tr: bool
    has_news: bool
    merged_return_files: List[str]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_20)
    ap.add_argument("--min-text-len", type=int, default=20)
    ap.add_argument("--min-tokens", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-operator", action="store_true", help="pass through to 31_extract_text_units.py")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--out-dir", type=str, default="data/_derived/text")
    ap.add_argument("--with-finbert", action="store_true", help="Also score FinBERT (transcripts + news).")
    ap.add_argument("--with-openai", action="store_true", help="Also score OpenAI (transcripts + news).")
    ap.add_argument(
        "--return-files",
        nargs="*",
        default=["event_abnormal_windows.csv", "event_window_returns.csv"],
        help="One or more per-ticker event-level return tables under data/<TICKER>/events/. All found are merged.",
    )
    ap.add_argument(
        "--corr-target",
        type=str,
        default=None,
        help="Return/abnormal-return column to use as correlation target (e.g., car_0_5). If omitted, auto-picks.",
    )
    ap.add_argument(
        "--corr-min-n",
        type=int,
        default=80,
        help="Minimum number of paired non-missing observations to compute a correlation.",
    )
    args = ap.parse_args()

    root = repo_root()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: extract units
    if not args.skip_extract:
        cmd = [
            sys.executable, "-u", str(root / "scripts" / "31_extract_text_units.py"),
            "--data-dir", str(data_dir),
            "--min-text-len", str(args.min_text_len),
            "--tickers", *args.tickers,
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.keep_operator:
            cmd.append("--keep-operator")
        run(cmd, cwd=root)

    # Step 2/3: score units with LM + aggregate event-level tables
    if not args.skip_score:
        cmd_tr = [
            sys.executable, "-u", str(root / "scripts" / "32_score_text_LM_transcripts.py"),
            "--data-dir", str(data_dir),
            "--min-tokens", str(args.min_tokens),
            "--tickers", *args.tickers,
        ]
        if args.overwrite:
            cmd_tr.append("--overwrite")
        run(cmd_tr, cwd=root)

        cmd_news = [
            sys.executable, "-u", str(root / "scripts" / "33_score_text_LM_news.py"),
            "--data-dir", str(data_dir),
            "--min-tokens", str(args.min_tokens),
            "--tickers", *args.tickers,
        ]
        if args.overwrite:
            cmd_news.append("--overwrite")
        run(cmd_news, cwd=root)

        if args.with_finbert:
            cmd_fb_tr = [
                sys.executable, "-u", str(root / "scripts" / "34_score_text_FinBERT_transcripts.py"),
                "--data-dir", str(data_dir),
                "--tickers", *args.tickers,
            ]
            if args.overwrite:
                cmd_fb_tr.append("--overwrite")
            run(cmd_fb_tr, cwd=root)

            cmd_fb_news = [
                sys.executable, "-u", str(root / "scripts" / "35_score_text_FinBERT_news.py"),
                "--data-dir", str(data_dir),
                "--tickers", *args.tickers,
            ]
            if args.overwrite:
                cmd_fb_news.append("--overwrite")
            run(cmd_fb_news, cwd=root)

        if args.with_openai:
            cmd_oa_tr = [
                sys.executable, "-u", str(root / "scripts" / "36_score_text_OpenAI_transcripts.py"),
                "--data-dir", str(data_dir),
                "--tickers", *args.tickers,
            ]
            if args.overwrite:
                cmd_oa_tr.append("--overwrite")
            run(cmd_oa_tr, cwd=root)

            cmd_oa_news = [
                sys.executable, "-u", str(root / "scripts" / "37_score_text_OpenAI_news.py"),
                "--data-dir", str(data_dir),
                "--tickers", *args.tickers,
            ]
            if args.overwrite:
                cmd_oa_news.append("--overwrite")
            run(cmd_oa_news, cwd=root)

    # Step 4: merge to panel
    all_rows: List[pd.DataFrame] = []
    metas: List[TickerMeta] = []

    KEY_CANDIDATES = [
        ("ticker", "earnings_date", "day0_date", "day0_dt"),
        ("ticker", "earnings_date", "day0_date"),
    ]

    for t in args.tickers:
        tdir = data_dir / t / "events"
        ev_path = tdir / "event_windows.csv"
        tr_path = tdir / "text_lm_transcripts_event_wide.csv"
        nw_path = tdir / "text_lm_news_event_wide.csv"
        fb_tr_path = tdir / "text_finbert_transcripts_event_wide.csv"
        fb_nw_path = tdir / "text_finbert_news_event_wide.csv"
        oa_tr_path = tdir / "text_oa_transcripts_event_wide.csv"
        oa_nw_path = tdir / "text_oa_news_event_wide.csv"

        ev = _read_csv_maybe(ev_path)
        if ev is None:
            print(f"[WARN] {t}: missing {ev_path}; skipping ticker", flush=True)
            continue

        tr = _read_csv_maybe(tr_path)
        nw = _read_csv_maybe(nw_path)
        fb_tr = _read_csv_maybe(fb_tr_path)
        fb_nw = _read_csv_maybe(fb_nw_path)
        oa_tr = _read_csv_maybe(oa_tr_path)
        oa_nw = _read_csv_maybe(oa_nw_path)

        # determine join keys available across tables
        keys = None
        for cand in KEY_CANDIDATES:
            if all(k in ev.columns for k in cand):
                ok_tr = (tr is None) or all(k in tr.columns for k in cand)
                ok_nw = (nw is None) or all(k in nw.columns for k in cand)
                if ok_tr and ok_nw:
                    keys = cand
                    break
        if keys is None:
            print(f"[WARN] {t}: could not find common join keys across tables; skipping", flush=True)
            continue

        ev = _ensure_str_cols(ev, keys)
        out = ev.copy()

        if tr is not None:
            tr = _ensure_str_cols(tr, keys)
            tr = tr.rename(columns={c: f"tr__{c}" for c in tr.columns if c not in keys})
            out = out.merge(tr, on=list(keys), how="left")
        if nw is not None:
            nw = _ensure_str_cols(nw, keys)
            nw = nw.rename(columns={c: f"news__{c}" for c in nw.columns if c not in keys})
            out = out.merge(nw, on=list(keys), how="left")
        if fb_tr is not None:
            fb_tr = _ensure_str_cols(fb_tr, keys)
            fb_tr = fb_tr.rename(columns={c: f"trfb__{c}" for c in fb_tr.columns if c not in keys})
            out = out.merge(fb_tr, on=list(keys), how="left")
        if fb_nw is not None:
            fb_nw = _ensure_str_cols(fb_nw, keys)
            fb_nw = fb_nw.rename(columns={c: f"newsfb__{c}" for c in fb_nw.columns if c not in keys})
            out = out.merge(fb_nw, on=list(keys), how="left")
        if oa_tr is not None:
            oa_tr = _ensure_str_cols(oa_tr, keys)
            oa_tr = oa_tr.rename(columns={c: f"troa__{c}" for c in oa_tr.columns if c not in keys})
            out = out.merge(oa_tr, on=list(keys), how="left")
        if oa_nw is not None:
            oa_nw = _ensure_str_cols(oa_nw, keys)
            oa_nw = oa_nw.rename(columns={c: f"newsoa__{c}" for c in oa_nw.columns if c not in keys})
            out = out.merge(oa_nw, on=list(keys), how="left")

        merged_ret_files: List[str] = []
        # Return/abnormal-return files often omit day0_date; pick the most specific join keys available.
        RET_KEY_CANDIDATES = [
            ("ticker", "earnings_date", "day0_date"),
            ("ticker", "earnings_date"),
            ("earnings_date", "day0_date"),
            ("earnings_date",),
        ]
        for rf in args.return_files:
            rf_path = tdir / rf
            ret = _read_csv_maybe(rf_path)
            if ret is None:
                continue

            # Many per-ticker return files omit ticker.
            if "ticker" not in ret.columns:
                ret["ticker"] = t

            # Normalize common column-name variants.
            lower_cols = {c.lower(): c for c in ret.columns}
            rename_map: Dict[str, str] = {}

            # earnings_date aliases
            if "earnings_date" not in ret.columns:
                for alias in ["earningsdate", "earnings_dt", "event_date", "event_dt", "date"]:
                    if alias in lower_cols:
                        rename_map[lower_cols[alias]] = "earnings_date"
                        break

            # day0_date aliases
            if "day0_date" not in ret.columns:
                for alias in ["day0date", "day0_dt", "day0dt", "day0"]:
                    if alias in lower_cols:
                        rename_map[lower_cols[alias]] = "day0_date"
                        break

            if rename_map:
                ret = ret.rename(columns=rename_map)

            # Pick join keys for this specific return file.
            ret_keys: Optional[List[str]] = None
            for cand in RET_KEY_CANDIDATES:
                if all(k in ret.columns for k in cand) and all(k in out.columns for k in cand):
                    ret_keys = list(cand)
                    break

            if ret_keys is None:
                print(f"[WARN] {t}: {rf} missing join keys; skipping")
                continue

            out = out.merge(ret, on=ret_keys, how="left")
            merged_ret_files.append(rf)


        # create a couple of helpful deltas/ratios (if columns exist)
        # News: pre vs post
        if "news__lm_sent_mean_pre" in out.columns and "news__lm_sent_mean_post" in out.columns:
            out["delta_news_sent_mean_post_minus_pre"] = out["news__lm_sent_mean_post"] - out["news__lm_sent_mean_pre"]
        if "news__lm_tone_mean_pre" in out.columns and "news__lm_tone_mean_post" in out.columns:
            out["delta_news_tone_mean_post_minus_pre"] = out["news__lm_tone_mean_post"] - out["news__lm_tone_mean_pre"]
        # Transcripts: prepared vs QA
        if "tr__lm_sent_mean_prepared" in out.columns and "tr__lm_sent_mean_qa" in out.columns:
            out["delta_tr_sent_mean_qa_minus_prepared"] = out["tr__lm_sent_mean_qa"] - out["tr__lm_sent_mean_prepared"]
        if "tr__lm_tone_mean_prepared" in out.columns and "tr__lm_tone_mean_qa" in out.columns:
            out["delta_tr_tone_mean_qa_minus_prepared"] = out["tr__lm_tone_mean_qa"] - out["tr__lm_tone_mean_prepared"]

        # attach ticker and event sequence index (1..N, per ticker sorted by date)
        out["ticker"] = t
        if "earnings_date" in out.columns:
            out = out.sort_values("earnings_date").reset_index(drop=True)
        out["event_seq"] = np.arange(1, len(out) + 1)

        all_rows.append(out)
        metas.append(TickerMeta(
            ticker=t,
            events=int(len(out)),
            has_tr=(tr is not None),
            has_news=(nw is not None),
            merged_return_files=merged_ret_files,
        ))

    if not all_rows:
        raise RuntimeError("No tickers produced output; check input paths and tickers list.")

    out_all = pd.concat(all_rows, ignore_index=True)
    out_all_path = out_dir / "event_text_features.csv"
    out_all.to_csv(out_all_path, index=False)

    # Summaries
    # Choose "interesting" numeric columns: text features + return targets + a few counts.
    num_cols = _numeric_cols(out_all)

    ret_targets = _detect_return_targets(out_all)
    text_feat_cols = [c for c in num_cols if _is_text_feature_col(c)]
    # also include return targets in summaries
    summary_cols = sorted(set(text_feat_cols + ret_targets))

    def _group_mean(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        g = df.groupby(group_col)[summary_cols].mean(numeric_only=True)
        g.insert(0, "n_events", df.groupby(group_col).size())
        return g.reset_index()

    summary_by_ticker = _group_mean(out_all, "ticker")
    summary_by_event_seq = _group_mean(out_all, "event_seq")

    summary_by_ticker_path = out_dir / "summary_by_ticker.csv"
    summary_by_event_seq_path = out_dir / "summary_by_event_seq.csv"
    summary_by_ticker.to_csv(summary_by_ticker_path, index=False)
    summary_by_event_seq.to_csv(summary_by_event_seq_path, index=False)

    # Correlations
    y_col = _pick_corr_target(out_all, args.corr_target)
    corr_target_found = (y_col is not None)

    top_corr_path = out_dir / "top_correlations.csv"
    top_corr_by_target_path = out_dir / "top_correlations_by_target.csv"

    if corr_target_found:
        top_corr = _top_correlations(out_all, y_col=y_col, candidate_cols=text_feat_cols, min_n=args.corr_min_n)
        top_corr.to_csv(top_corr_path, index=False)
    else:
        top_corr = pd.DataFrame(columns=["feature", "corr", "n"])
        top_corr.to_csv(top_corr_path, index=False)

    # correlations for all targets (long format)
    long_rows = []
    for tgt in ret_targets:
        tc = _top_correlations(out_all, y_col=tgt, candidate_cols=text_feat_cols, min_n=args.corr_min_n)
        if tc.empty:
            continue
        tc = tc.head(50).copy()
        tc.insert(0, "target", tgt)
        long_rows.append(tc)
    if long_rows:
        corr_by_target = pd.concat(long_rows, ignore_index=True)
    else:
        corr_by_target = pd.DataFrame(columns=["target", "feature", "corr", "n"])
    corr_by_target.to_csv(top_corr_by_target_path, index=False)

    # Meta snapshot
    meta = {
        "tickers": args.tickers,
        "rows": int(out_all.shape[0]),
        "cols": int(out_all.shape[1]),
        "out_path": str(out_all_path),
        "return_files_requested": args.return_files,
        "return_targets_detected": ret_targets,
        "corr_target": y_col,
        "per_ticker": [
            {
                "ticker": m.ticker,
                "events": m.events,
                "tr": m.has_tr,
                "news": m.has_news,
                "merged_return_files": m.merged_return_files,
            }
            for m in metas
        ],
    }
    (out_dir / "event_text_features.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Console summary (compact)
    print("\n=== [TEXT PIPELINE SUMMARY] ===", flush=True)
    print(f"tickers={len(metas)} rows={out_all.shape[0]} cols={out_all.shape[1]}", flush=True)
    for m in metas:
        has_ret = bool(m.merged_return_files)
        print(f"[TICKER] {m.ticker}: events={m.events} tr={m.has_tr} news={m.has_news} ret={has_ret}", flush=True)

    if corr_target_found and not top_corr.empty:
        print(f"\n[OK] correlation target: {y_col}", flush=True)
        print("[TOP] abs(corr) text-features vs target:", flush=True)
        for _, r in top_corr.head(8).iterrows():
            print(f"  {r['feature']}: corr={r['corr']:.3f} (n={int(r['n'])})", flush=True)
    else:
        print("\n[NOTE] no return/abnormal-return target column found; correlations skipped", flush=True)

    print(f"\n[OK] wrote merged event text features -> {out_all_path} rows={out_all.shape[0]} cols={out_all.shape[1]}", flush=True)
    print(f"[OK] wrote summaries -> {summary_by_ticker_path.name}, {summary_by_event_seq_path.name}, {top_corr_path.name}, {top_corr_by_target_path.name}", flush=True)


if __name__ == "__main__":
    main()
