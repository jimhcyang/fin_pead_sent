#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/31_extract_text_units.py

Extract "text units" for:
  1) News: from data/{TICKER}/news/**/stock_news.csv (stable endpoint outputs)
  2) Earnings-call transcripts: from data/{TICKER}/transcripts/YYYY-MM-DD/transcript.txt

Outputs:
  data/{TICKER}/events/text_units_news.csv
  data/{TICKER}/events/text_units_transcripts.csv

Key changes (2026-02-06):
  - Q&A split is more robust using your 2-line Operator/first-question rules.
  - Windows for news + transcript alignment are centered on the earnings CALL date
    (transcript folder date), not necessarily the earnings release/document date.
  - We map each event in events/event_windows.csv to the nearest transcript call date
    within a tolerance (default 7 days), so tickers where call != release date no longer
    "miss transcripts" due to folder mismatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

from _common import parse_dt_any, to_et

ET = ZoneInfo("America/New_York")

DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# -------------------------
# Data structures
# -------------------------
@dataclass
class EventWindow:
    ticker: str
    earnings_date: str   # event key from event_windows.csv (often earnings release date)
    day0_date: str       # event key (often trading-day anchor)
    day0_dt: Optional[str]
    dt_m5: Optional[str]
    dt_p10: Optional[str]

    # New / derived
    call_date: Optional[str] = None          # transcript folder date
    call_dt: Optional[str] = None            # transcript timestamp if available
    window_center_date: Optional[str] = None # call_date if exists else earnings_date


# -------------------------
# Helpers
# -------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _dt_iso_at(d: date, hh: int, mm: int, ss: int = 0) -> str:
    return datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=ET).isoformat()


def shift_weekdays(d: date, n: int) -> date:
    """Shift by n weekdays (Mon–Fri). Holidays are NOT excluded."""
    if n == 0:
        return d
    step = 1 if n > 0 else -1
    cur = d
    for _ in range(abs(int(n))):
        cur = cur + timedelta(days=step)
        while cur.weekday() >= 5:
            cur = cur + timedelta(days=step)
    return cur


def _list_transcript_dates(transcripts_dir: Path) -> List[str]:
    if not transcripts_dir.exists():
        return []
    out: List[str] = []
    for child in transcripts_dir.iterdir():
        if child.is_dir() and DATE_DIR_RE.match(child.name):
            out.append(child.name)
    return sorted(out)


def _load_transcript_datetime(transcript_dir: Path) -> Optional[str]:
    """
    If transcript.json exists and has a 'date' field, parse it to ET and return iso.
    Otherwise None.
    """
    jpath = transcript_dir / "transcript.json"
    if not jpath.exists():
        return None
    try:
        obj = json.loads(jpath.read_text(encoding="utf-8"))
        raw = str(obj.get("date", "")).strip()
        if not raw:
            return None
        dt = parse_dt_any(raw, assume_tz=ET)
        dt_et = to_et(dt)
        return dt_et.isoformat()
    except Exception:
        return None


def _nearest_date_match(
    target: str,
    candidates: List[str],
    used: set[str],
    tolerance_days: int,
) -> Optional[str]:
    """
    Find nearest unused candidate date to target within tolerance.
    """
    try:
        t = _parse_yyyy_mm_dd(target)
    except Exception:
        return None

    best: Optional[Tuple[int, str]] = None  # (abs_days, date_str)
    for c in candidates:
        if c in used:
            continue
        try:
            cd = _parse_yyyy_mm_dd(c)
        except Exception:
            continue
        diff = abs((cd - t).days)
        if diff <= tolerance_days:
            if best is None or diff < best[0]:
                best = (diff, c)

    return best[1] if best else None


# -------------------------
# Load Events
# -------------------------
def load_event_windows(events_csv: Path, ticker: str) -> List[EventWindow]:
    if not events_csv.exists():
        raise FileNotFoundError(f"Missing events file: {events_csv}")

    df = pd.read_csv(events_csv, dtype=str, keep_default_na=False)
    if df.empty:
        return []

    # Required fields
    required = {"ticker", "earnings_date", "day0_date"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{events_csv} missing required columns: {sorted(missing)}")

    # Optional dt columns
    for c in ["day0_dt", "dt_m5", "dt_p10"]:
        if c not in df.columns:
            df[c] = ""

    out: List[EventWindow] = []
    for _, r in df.iterrows():
        if str(r.get("ticker", "")).strip().upper() != ticker.upper():
            continue
        out.append(
            EventWindow(
                ticker=ticker.upper(),
                earnings_date=str(r.get("earnings_date", "")).strip(),
                day0_date=str(r.get("day0_date", "")).strip(),
                day0_dt=(str(r.get("day0_dt", "")).strip() or None),
                dt_m5=(str(r.get("dt_m5", "")).strip() or None),
                dt_p10=(str(r.get("dt_p10", "")).strip() or None),
            )
        )

    # Sort ascending by earnings_date as a stable baseline
    out.sort(key=lambda e: e.earnings_date)
    return out


def attach_call_dates(
    events: List[EventWindow],
    transcripts_dir: Path,
    tolerance_days: int,
    pre_bdays: int,
    post_bdays: int,
) -> None:
    """
    Mutates EventWindow objects:
      - call_date: matched transcript folder date
      - call_dt: parsed datetime from transcript.json if available
      - window_center_date: call_date if exists else earnings_date
      - dt_m5/dt_p10: recomputed around window_center_date (call-centered)
      - day0_dt: if call_dt exists, use that for pre/post phase split; else keep prior
    """
    tx_dates = _list_transcript_dates(transcripts_dir)
    used: set[str] = set()

    for ev in events:
        # 1) Exact folder match first
        if (transcripts_dir / ev.earnings_date).exists():
            ev.call_date = ev.earnings_date
            used.add(ev.call_date)
        else:
            # 2) Nearest transcript folder date within tolerance
            m = _nearest_date_match(ev.earnings_date, tx_dates, used, tolerance_days=tolerance_days)
            if m:
                ev.call_date = m
                used.add(m)
            else:
                ev.call_date = None

        # Decide center date
        center = ev.call_date or ev.earnings_date
        ev.window_center_date = center

        # Load call datetime if possible
        if ev.call_date:
            ev.call_dt = _load_transcript_datetime(transcripts_dir / ev.call_date)
        else:
            ev.call_dt = None

        # Compute window start/end around center date using business-day shifting
        try:
            d0 = _parse_yyyy_mm_dd(center)
            d_start = shift_weekdays(d0, -int(pre_bdays))
            d_end = shift_weekdays(d0, int(post_bdays))
            ev.dt_m5 = _dt_iso_at(d_start, 0, 0, 0)
            ev.dt_p10 = _dt_iso_at(d_end, 23, 59, 59)
        except Exception:
            # keep old values if parsing fails
            pass

        # day0_dt for phase split: prefer call_dt
        if ev.call_dt:
            ev.day0_dt = ev.call_dt
        else:
            # If none provided, set noon ET at center date
            if not ev.day0_dt:
                try:
                    dd = _parse_yyyy_mm_dd(center)
                    ev.day0_dt = _dt_iso_at(dd, 12, 0, 0)
                except Exception:
                    ev.day0_dt = None


# -------------------------
# Load News (all folders)
# -------------------------
def load_all_news(news_root: Path, ticker: str) -> pd.DataFrame:
    if not news_root.exists():
        return pd.DataFrame()

    files = list(news_root.glob("**/stock_news.csv"))
    if not files:
        # also support older naming if you used explicit range mode
        files = list(news_root.glob("**/stock_news*.csv"))

    rows: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp, dtype=str, keep_default_na=False)
            if not df.empty:
                rows.append(df)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["ticker"] = ticker.upper()
    return out


# -------------------------
# Transcript splitting (robust Q&A)
# -------------------------
_QNA_HEADINGS = (
    "questions and answers",
    "question and answer",
    "question-and-answer",
    "q&a",
    "q & a",
    "q and a",
    "questions & answers",
    "questions and answer session",
    "question-and-answer session",
)


def split_transcript_into_units(
    transcript_text: str,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    """
    Returns:
      units: list of (section, qa_marker, line_text)
      section_counts: prepared/qa counts
    """

    lines_raw = transcript_text.splitlines()
    lines = [ln.strip() for ln in lines_raw]
    # keep non-empty only for first-line logic, but still iterate full list
    first_non_empty_idx = None
    for i, ln in enumerate(lines):
        if ln.strip():
            first_non_empty_idx = i
            break

    first_line_is_operator = False
    if first_non_empty_idx is not None:
        first_line_is_operator = lines[first_non_empty_idx].lower().startswith("operator:")

    section = "prepared"
    qa_marker = ""
    operator_count = 0
    prev_lower = ""

    units: List[Tuple[str, str, str]] = []
    counts = {"prepared": 0, "qa": 0}

    for ln in lines:
        s = ln.strip()
        if not s:
            prev_lower = ""
            continue

        low = s.lower()

        # Already in QA → keep QA
        if section != "qa":
            # heading-based QA
            if any(h in low for h in _QNA_HEADINGS):
                section = "qa"
                qa_marker = "qna_heading"

            # explicit "Q and A" short line
            elif low in {"q&a", "q & a", "q and a"}:
                section = "qa"
                qa_marker = "q_and_a"

            # question labels
            elif low.startswith("question:") or re.match(r"^q[\.:]\s*", low):
                section = "qa"
                qa_marker = "question"

            # OPERATOR rules (your robust logic)
            elif low.startswith("operator:"):
                operator_count += 1

                # Rule 2: first question context (same or previous line)
                if "first question" in low:
                    section = "qa"
                    qa_marker = "operator_contains_first_question"
                elif "first question" in prev_lower:
                    section = "qa"
                    qa_marker = "operator_prevline_contains_first_question"

                # Rule 1: second operator when first non-empty line is operator
                elif first_line_is_operator and operator_count >= 2:
                    section = "qa"
                    qa_marker = "operator_second_after_first_line_operator"

                # Extra robustness: second operator even if first line isn't operator
                elif operator_count >= 2:
                    section = "qa"
                    qa_marker = "operator_second"

        units.append((section, qa_marker, s))
        counts[section] += 1
        prev_lower = low

    return units, counts


# -------------------------
# Build transcript units per event
# -------------------------
def build_transcript_units(
    events: List[EventWindow],
    data_dir: Path,
    min_text_len: int,
) -> Tuple[pd.DataFrame, int, Dict[str, int], Dict[str, int]]:
    """
    Returns:
      df_units
      missing_transcript_events
      section_counts_total
      qa_marker_counts
    """
    rows: List[Dict[str, Any]] = []
    missing = 0
    section_totals = {"prepared": 0, "qa": 0}
    qa_marker_counts: Dict[str, int] = {}

    for ev in events:
        tx_root = data_dir / ev.ticker / "transcripts"

        # Prefer call_date folder if present; else fallback to earnings_date
        folder_date = None
        if ev.call_date and (tx_root / ev.call_date / "transcript.txt").exists():
            folder_date = ev.call_date
        elif (tx_root / ev.earnings_date / "transcript.txt").exists():
            folder_date = ev.earnings_date

        if not folder_date:
            missing += 1
            continue

        tpath = tx_root / folder_date / "transcript.txt"
        try:
            text = tpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            missing += 1
            continue

        units, counts = split_transcript_into_units(text)
        section_totals["prepared"] += counts.get("prepared", 0)
        section_totals["qa"] += counts.get("qa", 0)

        for (sec, marker, ln) in units:
            if len(ln) < min_text_len:
                continue
            if sec == "qa":
                qa_marker_counts[marker or ""] = qa_marker_counts.get(marker or "", 0) + 1

            rows.append(
                {
                    "ticker": ev.ticker,
                    "earnings_date": ev.earnings_date,
                    "day0_date": ev.day0_date,
                    "day0_dt": ev.day0_dt or "",
                    "call_date": ev.call_date or "",
                    "transcript_folder_date": folder_date,
                    "window_center_date": ev.window_center_date or "",
                    "section": sec,
                    "qa_marker": marker,
                    "text": ln,
                }
            )

    df = pd.DataFrame(rows)
    return df, missing, section_totals, qa_marker_counts


# -------------------------
# Build news units per event (call-centered)
# -------------------------
def _event_center_datetime(ev: EventWindow) -> Optional[datetime]:
    if ev.day0_dt:
        try:
            dt = parse_dt_any(ev.day0_dt, assume_tz=ET)
            return to_et(dt)
        except Exception:
            return None
    return None


def build_news_units(
    events: List[EventWindow],
    news_df: pd.DataFrame,
    min_text_len: int,
) -> Tuple[pd.DataFrame, int]:
    """
    Filters news_df into event windows per event:
      [dt_m5, dt_p10] computed around call_date if available.
    Also labels pre/post relative to call datetime (day0_dt updated to call_dt if available).
    """
    if news_df.empty:
        return pd.DataFrame(), len(events)

    # normalize to datetime
    if "publishedDateET" not in news_df.columns:
        # attempt fallback
        news_df["publishedDateET"] = news_df.get("publishedDate", "")

    def _parse_news_dt(x: str) -> Optional[datetime]:
        x = str(x or "").strip()
        if not x:
            return None
        try:
            dt = parse_dt_any(x, assume_tz=ET)
            return to_et(dt)
        except Exception:
            return None

    news_df = news_df.copy()
    news_df["_dt"] = news_df["publishedDateET"].apply(_parse_news_dt)
    news_df = news_df.dropna(subset=["_dt"]).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    missing = 0

    for ev in events:
        # must have window boundaries
        if not ev.dt_m5 or not ev.dt_p10:
            missing += 1
            continue

        try:
            w0 = parse_dt_any(ev.dt_m5, assume_tz=ET)
            w1 = parse_dt_any(ev.dt_p10, assume_tz=ET)
            w0 = to_et(w0)
            w1 = to_et(w1)
        except Exception:
            missing += 1
            continue

        call_dt = _event_center_datetime(ev)
        if call_dt is None:
            # fallback: center at noon of window center date
            try:
                dd = _parse_yyyy_mm_dd(ev.window_center_date or ev.earnings_date)
                call_dt = datetime(dd.year, dd.month, dd.day, 12, 0, tzinfo=ET)
            except Exception:
                call_dt = None

        sub = news_df[(news_df["_dt"] >= w0) & (news_df["_dt"] <= w1)]
        if sub.empty:
            # not “missing event”; just no news in window
            continue

        for _, r in sub.iterrows():
            txt = str(r.get("text", "") or "").strip()
            title = str(r.get("title", "") or "").strip()
            merged = (title + "\n" + txt).strip()
            if len(merged) < min_text_len:
                continue

            phase = "post"
            if call_dt and r["_dt"] < call_dt:
                phase = "pre"

            rows.append(
                {
                    "ticker": ev.ticker,
                    "earnings_date": ev.earnings_date,
                    "day0_date": ev.day0_date,
                    "day0_dt": ev.day0_dt or "",
                    "call_date": ev.call_date or "",
                    "window_center_date": ev.window_center_date or "",
                    "phase": phase,
                    "publishedDateET": r.get("publishedDateET", ""),
                    "site": r.get("site", ""),
                    "url": r.get("url", ""),
                    "title": title,
                    "text": merged,
                }
            )

    df = pd.DataFrame(rows)
    return df, missing


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--min-text-len", type=int, default=20)

    # call-centered window options
    ap.add_argument("--pre-bdays", type=int, default=5)
    ap.add_argument("--post-bdays", type=int, default=10)
    ap.add_argument("--call-match-tolerance-days", type=int, default=7)

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    min_text_len = int(args.min_text_len)

    for t in args.tickers:
        ticker = t.strip().upper()
        print(f"\n=== [TEXT UNITS] {ticker} ===", flush=True)

        events_csv = data_dir / ticker / "events" / "event_windows.csv"
        events = load_event_windows(events_csv, ticker)

        # Attach call dates and recompute windows call-centered
        transcripts_dir = data_dir / ticker / "transcripts"
        attach_call_dates(
            events,
            transcripts_dir=transcripts_dir,
            tolerance_days=int(args.call_match_tolerance_days),
            pre_bdays=int(args.pre_bdays),
            post_bdays=int(args.post_bdays),
        )

        # Load news
        news_root = data_dir / ticker / "news"
        news_df = load_all_news(news_root, ticker)

        # Build units
        df_news, missing_news_events = build_news_units(events, news_df, min_text_len=min_text_len)
        df_tx, missing_tx_events, sec_counts, qa_marker_counts = build_transcript_units(
            events, data_dir=data_dir, min_text_len=min_text_len
        )

        # Write outputs
        out_dir = data_dir / ticker / "events"
        _ensure_dir(out_dir)

        news_path = out_dir / "text_units_news.csv"
        tx_path = out_dir / "text_units_transcripts.csv"

        df_news.to_csv(news_path, index=False)
        df_tx.to_csv(tx_path, index=False)

        print(f"[OK] wrote news units       -> {news_path}  rows={len(df_news):,}", flush=True)
        print(f"[OK] wrote transcript units -> {tx_path}  rows={len(df_tx):,}", flush=True)

        print(
            f"[NOTE] events_total={len(events)} missing_news_events={missing_news_events} missing_transcript_events={missing_tx_events}",
            flush=True,
        )
        print(
            f"[NOTE] transcript sections: prepared={sec_counts.get('prepared',0)} qa={sec_counts.get('qa',0)}",
            flush=True,
        )

        if qa_marker_counts:
            top = sorted(qa_marker_counts.items(), key=lambda kv: kv[1], reverse=True)[0]
            print(f"[NOTE] top qa_marker: {top[0]}  count={top[1]}", flush=True)
        else:
            print("[NOTE] no qa_marker counts (no QA lines or no transcripts)", flush=True)


if __name__ == "__main__":
    main()
