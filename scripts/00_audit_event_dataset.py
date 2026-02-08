#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_audit_event_dataset.py

Audit event + text pipeline coverage and missingness.

What it produces (under data/_derived/audits/<run_id>/):
- audit_events.csv
    Per ticker × event_date:
      - text_units coverage (news: pre/post; transcripts: prepared/qa)
      - LM wide coverage (news + transcripts)
      - (optional) identifier fields merged from event_features_stable / event_panel if present
- audit_news_publishers_by_event_phase.csv
    Publisher counts by ticker/event/phase (pre/post) from text_units_news.
- audit_transcript_lengths_by_event_section.csv
    Token/line summaries by ticker/event/section (prepared/qa) from text_units_transcripts.
- audit_missingness_by_column.csv
    Non-null counts for every column in key per-event tables (panel, stable features, LM wide, etc).
- audit_missing_cells_detail.csv
    Exact (table, ticker, event_date/event_seq, column) rows that are missing for columns with *small* missingness.
- audit_diff_coverage.csv
    For each *_post_minus_pre feature: how many events have missing pre vs missing post vs both.

Usage:
  python scripts/00_audit_event_dataset.py --data-dir data
  python scripts/00_audit_event_dataset.py --data-dir data --tickers AAPL MSFT

Notes:
- We DO NOT audit “raw news/transcripts exist” (script 07 already validated that).
- Token counts use tiktoken if available; otherwise fall back to a robust regex tokenization.
- Resilient to slightly different column names.
"""

import argparse
import datetime as dt
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# helpers: logging + paths
# -----------------------------

def _now_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def read_csv_safe(path: Path, logger: logging.Logger, warn_missing: bool = True) -> Optional[pd.DataFrame]:
    if not path.exists():
        if warn_missing:
            logger.warning(f"missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"failed reading {path}: {e}")
        return None


def choose_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # allow case-insensitive exact match
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None


def as_date_str(x) -> Optional[str]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s
        return pd.to_datetime(s).date().isoformat()
    except Exception:
        return None


# -----------------------------
# token counting
# -----------------------------

_TOKENIZER = None

def _init_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return
    try:
        import tiktoken  # type: ignore
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKENIZER = False


def count_tokens(text: str) -> int:
    if text is None:
        return 0
    s = str(text)
    if s == "" or s.lower() == "nan":
        return 0
    _init_tokenizer()
    if _TOKENIZER and _TOKENIZER is not False:
        try:
            return len(_TOKENIZER.encode(s))
        except Exception:
            pass
    return len(re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE))


def count_lines(text: str) -> int:
    if text is None:
        return 0
    s = str(text)
    if s == "" or s.lower() == "nan":
        return 0
    return s.count("\n") + 1


# -----------------------------
# ticker discovery
# -----------------------------

def discover_tickers(data_dir: Path) -> List[str]:
    tickers = []
    for p in data_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith("_"):
            continue
        if not re.fullmatch(r"[A-Z0-9\.\-]{1,10}", name):
            continue
        if (p / "events").exists():
            tickers.append(name)
    return sorted(tickers)


# -----------------------------
# audit per ticker
# -----------------------------

def audit_ticker(
    ticker: str,
    data_dir: Path,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    base = data_dir / ticker
    events_dir = base / "events"

    # core event list
    windows_path = events_dir / "event_windows.csv"
    df_win = read_csv_safe(windows_path, logger, warn_missing=True)
    if df_win is None or df_win.empty:
        logger.warning(f"{ticker}: no event_windows.csv; skipping ticker")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})

    event_date_col = choose_col(df_win, ["event_date", "earnings_date", "date"])
    if event_date_col is None:
        logger.error(f"{ticker}: cannot find event date column in {windows_path.name}")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})

    df_win[event_date_col] = df_win[event_date_col].apply(as_date_str)
    df_win = df_win.dropna(subset=[event_date_col]).copy()
    df_win = df_win.sort_values(event_date_col).reset_index(drop=True)
    df_win["event_seq"] = np.arange(1, len(df_win) + 1, dtype=int)

    # per-event skeleton
    df_evt = df_win[["event_seq", event_date_col]].rename(columns={event_date_col: "event_date"}).copy()
    df_evt.insert(0, "ticker", ticker)

    # -------------------
    # text_units_news: coverage, publishers, tokens
    # -------------------
    pub_by_event_phase: List[pd.DataFrame] = []
    news_units_summary: List[pd.DataFrame] = []

    tu_news_path = events_dir / "text_units_news.csv"
    df_tun = read_csv_safe(tu_news_path, logger, warn_missing=True)
    if df_tun is not None and not df_tun.empty:
        ed_col = choose_col(df_tun, ["event_date", "earnings_date", "date"])
        phase_col = choose_col(df_tun, ["phase", "news_phase"])
        text_col = choose_col(df_tun, ["text", "content", "unit_text", "body"])
        pub_col = choose_col(df_tun, ["publisher", "site", "source", "publication", "domain"])

        if ed_col:
            df_tun[ed_col] = df_tun[ed_col].apply(as_date_str)

        if ed_col and phase_col:
            if text_col and text_col in df_tun.columns:
                df_tun["_tokens"] = df_tun[text_col].apply(count_tokens)
                df_tun["_lines"] = df_tun[text_col].apply(count_lines)
                df_tun["_chars"] = df_tun[text_col].astype(str).str.len()
            else:
                df_tun["_tokens"] = 0
                df_tun["_lines"] = 0
                df_tun["_chars"] = 0

            g = df_tun.groupby([ed_col, phase_col], dropna=False)
            sum_df = g.agg(
                news_units=(phase_col, "size"),
                news_tokens=("_tokens", "sum"),
                news_lines=("_lines", "sum"),
                news_chars=("_chars", "sum"),
            ).reset_index().rename(columns={ed_col: "event_date", phase_col: "phase"})

            if pub_col and pub_col in df_tun.columns:
                u = df_tun.groupby([ed_col, phase_col])[pub_col].nunique().reset_index()
                u = u.rename(columns={ed_col: "event_date", phase_col: "phase", pub_col: "news_unique_publishers_units"})
                sum_df = sum_df.merge(u, on=["event_date", "phase"], how="left")

                pc = df_tun.groupby([ed_col, phase_col, pub_col]).size().reset_index(name="n_units")
                pc = pc.rename(columns={ed_col: "event_date", phase_col: "phase", pub_col: "publisher"})
                pc.insert(0, "ticker", ticker)
                pub_by_event_phase.append(pc)

            sum_df.insert(0, "ticker", ticker)
            news_units_summary.append(sum_df)

    if news_units_summary:
        df_news_sum = pd.concat(news_units_summary, ignore_index=True)

        values = ["news_units", "news_tokens", "news_lines", "news_chars"]
        if "news_unique_publishers_units" in df_news_sum.columns:
            values.append("news_unique_publishers_units")

        piv = df_news_sum.pivot_table(
            index=["ticker", "event_date"],
            columns="phase",
            values=values,
            aggfunc="first",
        )
        piv.columns = [f"units_news__{m}__{str(ph)}" for (m, ph) in piv.columns]
        piv = piv.reset_index()
        df_evt = df_evt.merge(piv, on=["ticker", "event_date"], how="left")

    df_pub = (
        pd.concat(pub_by_event_phase, ignore_index=True)
        if pub_by_event_phase
        else pd.DataFrame(columns=["ticker", "event_date", "phase", "publisher", "n_units"])
    )

    # -------------------
    # text_units_transcripts: prepared vs qa breakdown
    # -------------------
    tr_section_summary: List[pd.DataFrame] = []
    tu_tr_path = events_dir / "text_units_transcripts.csv"
    df_tut = read_csv_safe(tu_tr_path, logger, warn_missing=True)
    if df_tut is not None and not df_tut.empty:
        ed_col = choose_col(df_tut, ["event_date", "earnings_date", "date"])
        sec_col = choose_col(df_tut, ["section", "tr_section", "transcript_section"])
        text_col = choose_col(df_tut, ["text", "content", "unit_text", "body"])

        if ed_col:
            df_tut[ed_col] = df_tut[ed_col].apply(as_date_str)

        if text_col and text_col in df_tut.columns:
            df_tut["_tokens"] = df_tut[text_col].apply(count_tokens)
            df_tut["_lines"] = df_tut[text_col].apply(count_lines)
            df_tut["_chars"] = df_tut[text_col].astype(str).str.len()
        else:
            df_tut["_tokens"] = 0
            df_tut["_lines"] = 0
            df_tut["_chars"] = 0

        if ed_col and sec_col:
            g = df_tut.groupby([ed_col, sec_col], dropna=False)
            sum_df = g.agg(
                tr_units=(sec_col, "size"),
                tr_tokens=("_tokens", "sum"),
                tr_lines=("_lines", "sum"),
                tr_chars=("_chars", "sum"),
            ).reset_index().rename(columns={ed_col: "event_date", sec_col: "section"})
            sum_df.insert(0, "ticker", ticker)
            tr_section_summary.append(sum_df)

    df_trsec = (
        pd.concat(tr_section_summary, ignore_index=True)
        if tr_section_summary
        else pd.DataFrame(columns=["ticker", "event_date", "section", "tr_units", "tr_tokens", "tr_lines", "tr_chars"])
    )

    if not df_trsec.empty:
        piv = df_trsec.pivot_table(
            index=["ticker", "event_date"],
            columns="section",
            values=["tr_units", "tr_tokens", "tr_lines", "tr_chars"],
            aggfunc="first",
        )
        piv.columns = [f"units_tr__{m}__{str(sec)}" for (m, sec) in piv.columns]
        piv = piv.reset_index()
        df_evt = df_evt.merge(piv, on=["ticker", "event_date"], how="left")

    # -------------------
    # LM wide files: missingness signals + merge into df_evt
    # -------------------
    extra_tables: Dict[str, pd.DataFrame] = {}

    def merge_lm_wide(fname: str, prefix: str):
        p = events_dir / fname
        df = read_csv_safe(p, logger, warn_missing=True)
        if df is None or df.empty:
            return
        ed_col = choose_col(df, ["event_date", "earnings_date", "date"])
        if ed_col is None:
            return
        df[ed_col] = df[ed_col].apply(as_date_str)
        df2 = df.rename(columns={ed_col: "event_date"}).copy()
        if "ticker" not in df2.columns:
            df2.insert(0, "ticker", ticker)

        extra_tables[f"{prefix}_wide"] = df2

        df_evt_local = df2.copy()
        # prefix non-id cols for merge into df_evt
        ren = {}
        for c in df_evt_local.columns:
            if c in ("ticker", "event_date"):
                continue
            ren[c] = f"{prefix}__{c}"
        df_evt_local = df_evt_local.rename(columns=ren)

        nonlocal df_evt
        df_evt = df_evt.merge(df_evt_local, on=["ticker", "event_date"], how="left")

    merge_lm_wide("text_lm_news_event_wide.csv", "lm_news")
    merge_lm_wide("text_lm_transcripts_event_wide.csv", "lm_tr")

    # -------------------
    # stable features & panel: store for missingness + optionally merge identifiers
    # -------------------
    for fname, key in [
        ("event_features_stable.csv", "stable"),
        ("event_panel.csv", "panel"),
    ]:
        p = events_dir / fname
        df = read_csv_safe(p, logger, warn_missing=True)
        if df is None or df.empty:
            continue

        ed_col = choose_col(df, ["event_date", "earnings_date", "date"])
        if ed_col:
            df[ed_col] = df[ed_col].apply(as_date_str)
            df2 = df.rename(columns={ed_col: "event_date"}).copy()
        else:
            df2 = df.copy()

        if "ticker" not in df2.columns:
            df2.insert(0, "ticker", ticker)

        extra_tables[key] = df2

        # merge a few identifier-ish columns into df_evt (not the full feature set)
        merge_cols = ["ticker"]
        if "event_date" in df2.columns:
            merge_cols.append("event_date")
        elif "event_seq" in df2.columns and "event_seq" in df_evt.columns:
            merge_cols.append("event_seq")
        else:
            continue

        id_cols = [c for c in [
            "event_seq", "event_date",
            "fiscalDateEnding", "reportDate", "date", "period",
            "calendar_date", "fiscal_period"
        ] if c in df2.columns]

        tmp = df2[["ticker"] + [c for c in id_cols if c != "ticker"]].drop_duplicates()
        df_evt = df_evt.merge(tmp, on=merge_cols, how="left")

    logger.info(f"{ticker}: events={len(df_evt)}")
    return df_evt, df_pub, df_trsec, extra_tables


# -----------------------------
# missingness + diff diagnostics
# -----------------------------

def missingness_by_column(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        if df is None or df.empty:
            continue
        n = len(df)
        for c in df.columns:
            nn = int(df[c].notna().sum())
            rows.append({
                "table": name,
                "column": c,
                "rows": n,
                "non_null": nn,
                "missing": n - nn,
                "missing_rate": (n - nn) / n if n else np.nan,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["missing", "table", "column"], ascending=[False, True, True]).reset_index(drop=True)


def missing_cells_detail(df: pd.DataFrame, id_cols: List[str], max_missing_per_col: int = 50) -> pd.DataFrame:
    """
    For columns with 1..max_missing_per_col missing values, emit exact id rows.
    Keeps output manageable while still solving “399 vs 400”.
    """
    rows = []
    n = len(df)
    for c in df.columns:
        if c in id_cols:
            continue
        miss_mask = df[c].isna()
        m = int(miss_mask.sum())
        if m == 0 or m > max_missing_per_col:
            continue
        miss_df = df.loc[miss_mask, id_cols].copy()
        for _, r in miss_df.iterrows():
            rows.append({**{k: r[k] for k in id_cols}, "column": c, "missing_rows": m, "total_rows": n})
    return pd.DataFrame(rows)


def diff_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnose *_post_minus_pre columns by checking whether the corresponding *_post and *_pre exist.
    """
    rows = []
    cols = list(df.columns)
    diff_cols = [c for c in cols if ("post_minus_pre" in c)]
    for dcol in diff_cols:
        pre = dcol.replace("post_minus_pre", "pre")
        post = dcol.replace("post_minus_pre", "post")
        if pre not in df.columns or post not in df.columns:
            continue

        pre_na = df[pre].isna()
        post_na = df[post].isna()
        both = int((pre_na & post_na).sum())
        pre_only = int((pre_na & ~post_na).sum())
        post_only = int((~pre_na & post_na).sum())
        neither = int((~pre_na & ~post_na).sum())
        diff_nn = int(df[dcol].notna().sum())

        rows.append({
            "diff_col": dcol,
            "base_pre": pre,
            "base_post": post,
            "rows": int(len(df)),
            "diff_non_null": diff_nn,
            "missing_pre_only": pre_only,
            "missing_post_only": post_only,
            "missing_both": both,
            "have_both_pre_post": neither,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("diff_non_null").reset_index(drop=True)


# -----------------------------
# main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="path to data directory (e.g., data)")
    ap.add_argument("--tickers", nargs="*", default=None, help="optional tickers; otherwise auto-discover")
    ap.add_argument("--out-root", default=None, help="optional output root (default: <data-dir>/_derived/audits)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else (data_dir / "_derived" / "audits")
    run_id = _now_run_id()
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / "audit.log")
    logger.info(f"data_dir={data_dir}")
    logger.info(f"out_dir={out_dir}")

    tickers = args.tickers if args.tickers else discover_tickers(data_dir)
    if not tickers:
        logger.error("no tickers found")
        return

    all_events = []
    all_pub = []
    all_trsec = []

    # collect tables for missingness analysis
    tables_for_missingness: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        df_evt, df_pub, df_trsec, extra_tables = audit_ticker(t, data_dir, logger)

        if df_evt is not None and not df_evt.empty:
            all_events.append(df_evt)
        if df_pub is not None and not df_pub.empty:
            all_pub.append(df_pub)
        if df_trsec is not None and not df_trsec.empty:
            all_trsec.append(df_trsec)

        for k, v in extra_tables.items():
            if v is None or v.empty:
                continue
            tables_for_missingness[f"{t}__{k}"] = v

    df_all_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    df_all_pub = pd.concat(all_pub, ignore_index=True) if all_pub else pd.DataFrame()
    df_all_trsec = pd.concat(all_trsec, ignore_index=True) if all_trsec else pd.DataFrame()

    if not df_all_events.empty:
        df_all_events.to_csv(out_dir / "audit_events.csv", index=False)
        logger.info(f"wrote {out_dir / 'audit_events.csv'} rows={len(df_all_events)}")

    if not df_all_pub.empty:
        df_all_pub.to_csv(out_dir / "audit_news_publishers_by_event_phase.csv", index=False)
        logger.info(f"wrote {out_dir / 'audit_news_publishers_by_event_phase.csv'} rows={len(df_all_pub)}")

    if not df_all_trsec.empty:
        df_all_trsec.to_csv(out_dir / "audit_transcript_lengths_by_event_section.csv", index=False)
        logger.info(f"wrote {out_dir / 'audit_transcript_lengths_by_event_section.csv'} rows={len(df_all_trsec)}")

    miss = missingness_by_column(tables_for_missingness)
    if not miss.empty:
        miss.to_csv(out_dir / "audit_missingness_by_column.csv", index=False)
        logger.info(f"wrote {out_dir / 'audit_missingness_by_column.csv'} rows={len(miss)}")

    # detail rows for small-missing columns
    detail_frames = []
    for name, df in tables_for_missingness.items():
        if df is None or df.empty:
            continue
        id_cols = [c for c in ["ticker", "event_date", "event_seq", "fiscalDateEnding", "reportDate", "date", "period"] if c in df.columns]
        if "ticker" not in id_cols:
            continue
        if "event_date" not in id_cols and "event_seq" not in id_cols:
            continue
        det = missing_cells_detail(df, id_cols=id_cols, max_missing_per_col=50)
        if not det.empty:
            det.insert(0, "table", name)
            detail_frames.append(det)

    if detail_frames:
        df_det = pd.concat(detail_frames, ignore_index=True)
        df_det.to_csv(out_dir / "audit_missing_cells_detail.csv", index=False)
        logger.info(f"wrote {out_dir / 'audit_missing_cells_detail.csv'} rows={len(df_det)}")

    # diff coverage diagnosis (global)
    diff_frames = []
    for name, df in tables_for_missingness.items():
        if df is None or df.empty:
            continue
        if not any("post_minus_pre" in c for c in df.columns):
            continue
        dc = diff_coverage(df)
        if not dc.empty:
            dc.insert(0, "table", name)
            diff_frames.append(dc)

    if diff_frames:
        df_dc = pd.concat(diff_frames, ignore_index=True)
        df_dc.to_csv(out_dir / "audit_diff_coverage.csv", index=False)
        logger.info(f"wrote {out_dir / 'audit_diff_coverage.csv'} rows={len(df_dc)}")

    logger.info("DONE")


if __name__ == "__main__":
    main()
