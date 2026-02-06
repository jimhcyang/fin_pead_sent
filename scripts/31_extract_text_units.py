#!/usr/bin/env python3
# scripts/31_extract_text_units.py

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from _text_utils import (
    align_to_business_day,
    business_day_offset,
    detect_date_column,
    detect_text_column,
    parse_dt_any,
    safe_read_csv,
    safe_read_jsonl,
    split_transcript_sections,
)


def _uid(*parts: str) -> str:
    s = "|".join([p or "" for p in parts])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _read_earnings_dates(data_dir: Path, ticker: str) -> List[str]:
    cal = data_dir / ticker / "calendar" / "earnings_calendar.csv"
    df = safe_read_csv(cal)
    if df.empty:
        return []
    col = None
    for c in df.columns:
        if str(c).strip().lower() in {"earnings_date", "date"}:
            col = c
            break
    if col is None:
        return []
    out = []
    for x in df[col].tolist():
        if pd.isna(x):
            continue
        d = str(x).strip()
        if d:
            out.append(d)
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract text units (news articles + transcript sections) into standardized unit tables.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--out-dir", default="data/_derived/text")
    ap.add_argument("--pre-bdays", type=int, default=5)
    ap.add_argument("--post-bdays", type=int, default=10)
    ap.add_argument("--keep-text", action="store_true", help="Keep full text in output (large). Default: False.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = [t.upper() for t in (args.tickers or [])]
        if not tickers:
            # infer tickers from data dir
            tickers = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and len(p.name) <= 6])

    news_rows: List[Dict] = []
    tr_rows: List[Dict] = []

    missing_news_events = 0
    missing_tr_events = 0
    total_events = 0

    for tkr in tickers:
        eds = _read_earnings_dates(data_dir, tkr)
        for ed in eds:
            total_events += 1
            day0 = str(ed).strip()
            if not day0:
                continue

            # ---------- NEWS ----------
            news_csv = data_dir / tkr / "news" / day0 / "stock_news.csv"
            news_jsonl = data_dir / tkr / "news" / day0 / "stock_news.jsonl"
            df_news = safe_read_csv(news_csv)
            if df_news.empty and news_jsonl.exists():
                items = safe_read_jsonl(news_jsonl)
                df_news = pd.DataFrame(items)

            if df_news.empty:
                missing_news_events += 1
            else:
                text_col = detect_text_column(df_news)
                date_col = detect_date_column(df_news)

                for i, row in df_news.iterrows():
                    text = ""
                    if text_col and text_col in df_news.columns:
                        text = str(row.get(text_col) or "").strip()
                    if not text:
                        continue

                    # published date
                    pub_raw = None
                    if date_col and date_col in df_news.columns:
                        pub_raw = row.get(date_col)
                    ts = parse_dt_any(pub_raw)
                    if ts is None or pd.isna(ts):
                        # fall back to event date if unknown, but mark missing
                        pub_ymd_raw = day0
                    else:
                        pub_ymd_raw = ts.date().isoformat()

                    pub_ymd_bday = align_to_business_day(pub_ymd_raw, roll="forward")
                    bd_off = business_day_offset(day0, pub_ymd_bday)

                    # phase buckets (pre/event/post) within our event window
                    if bd_off < 0:
                        phase = "pre"
                    elif bd_off == 0:
                        phase = "event"
                    else:
                        phase = "post"

                    in_window = (-int(args.pre_bdays) <= bd_off <= int(args.post_bdays))

                    uid = _uid(tkr, day0, "news", str(i), str(row.get("url", "")), str(row.get("title", "")))

                    out = {
                        "ticker": tkr,
                        "earnings_date": day0,
                        "source": "news",
                        "unit_label": "article",
                        "unit_id": uid,
                        "unit_date_raw": pub_ymd_raw,
                        "unit_date_bday": pub_ymd_bday,
                        "bd_offset": int(bd_off),
                        "phase": phase,
                        "in_m5_p10": bool(in_window),
                        "title": str(row.get("title", "")).strip(),
                        "url": str(row.get("url", "")).strip(),
                    }
                    if args.keep_text:
                        out["text"] = text
                    else:
                        out["text"] = text  # keep for now; you can drop later if you want
                    news_rows.append(out)

            # ---------- TRANSCRIPTS ----------
            tr_txt = data_dir / tkr / "transcripts" / day0 / "transcript.txt"
            tr_json = data_dir / tkr / "transcripts" / day0 / "transcript.json"

            text_full = ""
            if tr_txt.exists():
                text_full = tr_txt.read_text(encoding="utf-8", errors="ignore")
            elif tr_json.exists():
                # minimal fallback: dump json as text
                text_full = tr_json.read_text(encoding="utf-8", errors="ignore")

            if not text_full.strip():
                missing_tr_events += 1
            else:
                sections = split_transcript_sections(text_full)
                for label, t in sections.items():
                    uid = _uid(tkr, day0, "transcript", label)
                    out = {
                        "ticker": tkr,
                        "earnings_date": day0,
                        "source": "transcript",
                        "unit_label": label,  # prepared / qa / full
                        "unit_id": uid,
                        "unit_date_raw": day0,
                        "unit_date_bday": day0,
                        "bd_offset": 0,
                        "phase": "event",
                        "in_m5_p10": True,
                    }
                    if args.keep_text:
                        out["text"] = t.strip()
                    else:
                        out["text"] = t.strip()
                    tr_rows.append(out)

    dfN = pd.DataFrame(news_rows)
    dfT = pd.DataFrame(tr_rows)

    out_news = out_dir / "text_units_news.csv"
    out_tr = out_dir / "text_units_transcript.csv"
    dfN.to_csv(out_news, index=False)
    dfT.to_csv(out_tr, index=False)

    print(f"[OK] wrote news units      -> {out_news}  rows={len(dfN):,}")
    print(f"[OK] wrote transcript units -> {out_tr}  rows={len(dfT):,}")
    print(f"[NOTE] events_total={total_events:,} missing_news_events={missing_news_events:,} missing_transcript_events={missing_tr_events:,}")


if __name__ == "__main__":
    main()
