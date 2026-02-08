#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/31_extract_text_units.py

Extract "text units" for:
  1) News: from data/{TICKER}/news/YYYY-MM-DD/stock_news.csv
  2) Earnings-call transcripts: from data/{TICKER}/transcripts/YYYY-MM-DD/transcript.txt

Outputs:
  data/{TICKER}/events/text_units_news.csv
  data/{TICKER}/events/text_units_transcripts.csv

Key behavior:
  - We align each earnings event to an earnings *call* date (transcript folder date).
    Alignment is 1-to-1 BY ORDER (sorted events vs sorted transcript folders) using the
    tail(min(N)) to avoid quarter drift when counts mismatch.
  - We label transcript units as section ∈ {prepared, qa} using a robust Q&A start detector:
      * lines 1–2 (non-empty) are ALWAYS ineligible to start Q&A (unless a hard marker overrides)
      * Q&A start must occur at/after line 4 and is usually within lines 4–10
      * we look for headings ("QUESTION-AND-ANSWER", "Q&A", etc.)
      * or a candidate speaker line ("Operator:" OR the same <NAME>: as transcript line 1)
        plus cues like "first question", "Q&A session", "[Operator Instructions]", etc.
  - For news, we attach each article to the event whose window is centered on the CALL date,
    and label phase ∈ {pre, post} relative to call datetime (from transcript.json if available;
    else noon ET on call_date).

Notes:
  - This script purposefully does NOT attempt a nearest-date matching between calendar and transcript.
    It is designed to be stable over time and avoid Q1/Q2 swaps when dates drift.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

from _common import parse_dt_any, to_et

ET = ZoneInfo("America/New_York")
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z .\-\'&]{0,80}):\s*(.*)$")
def _is_header_only(line: str) -> bool:
    """
    Returns True if the line has a colon and nothing after the first colon
    (e.g., 'John Doe - Firm:' or 'Operator:' with no trailing content).
    Used to drop leading roster/header lines.
    """
    if ":" not in line:
        return False
    before, after = line.split(":", 1)
    return before.strip() != "" and after.strip() == ""


# -------------------------
# Data structures
# -------------------------
@dataclass
class EventWindow:
    ticker: str
    earnings_date: str
    day0_date: str
    day0_dt: Optional[str] = None

    # attached call info
    call_date: Optional[str] = None
    call_dt: Optional[str] = None  # ISO ET if available
    call_gap_days: Optional[int] = None


# -------------------------
# Small utils
# -------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _log_warn(path: Path, msg: str) -> None:
    _ensure_dir(path.parent)
    ts = datetime.now(ET).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{ts} | {msg}\n")
    print(f"[WARN] {msg}", flush=True)


def choose_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None


def _parse_yyyy_mm_dd(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _list_transcript_dates(transcripts_dir: Path) -> List[str]:
    if not transcripts_dir.exists():
        return []
    out: List[str] = []
    for child in transcripts_dir.iterdir():
        if child.is_dir() and DATE_DIR_RE.match(child.name):
            out.append(child.name)
    return sorted(out)


def _load_transcript_datetime(transcript_dir: Path) -> Optional[str]:
    """If transcript.json exists and has a 'date' field, parse to ET ISO."""
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


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame()


def _detect_text_col(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    candidates = [
        "text",
        "content",
        "summary",
        "article",
        "body",
        "description",
        "snippet",
        "fullText",
        "full_text",
    ]
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None


def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    candidates = ["publishedDate", "published_date", "date", "datetime", "time"]
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None


# -------------------------
# Load events
# -------------------------

def load_event_windows(events_csv: Path, ticker: str) -> List[EventWindow]:
    if not events_csv.exists():
        raise FileNotFoundError(f"Missing events file: {events_csv}")

    df = pd.read_csv(events_csv, dtype=str, keep_default_na=False)
    if df.empty:
        return []

    required = {"ticker", "earnings_date", "day0_date"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{events_csv} missing required columns: {sorted(missing)}")

    if "day0_dt" not in df.columns:
        df["day0_dt"] = ""

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
            )
        )

    out.sort(key=lambda e: e.earnings_date)
    return out


# -------------------------
# Attach call dates (ORDERED 1-1)
# -------------------------

def attach_call_dates_ordered(
    events: List[EventWindow],
    transcripts_dir: Path,
    warn_gap_days: int = 45,
    strict_count: bool = False,
) -> None:
    """
    1-to-1 alignment BY ORDER (sorted):
      events sorted by earnings_date
      transcript folders sorted by YYYY-MM-DD

    If counts differ, align the tail min(N) of both (most recent quarters).
    """
    tx_dates = _list_transcript_dates(transcripts_dir)

    for ev in events:
        ev.call_date = None
        ev.call_dt = None
        ev.call_gap_days = None

    if not events or not tx_dates:
        return

    n_ev = len(events)
    n_tx = len(tx_dates)

    if n_ev != n_tx:
        msg = f"events={n_ev} transcripts={n_tx} (expected same). Aligning by tail(min) to avoid Q drift."
        if strict_count:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}", flush=True)

    n = min(n_ev, n_tx)
    ev_slice = events[-n:]
    tx_slice = tx_dates[-n:]

    for ev, call_date in zip(ev_slice, tx_slice):
        ev.call_date = call_date
        ev.call_dt = _load_transcript_datetime(transcripts_dir / call_date)

        # compute gap
        try:
            ed = _parse_yyyy_mm_dd(ev.earnings_date)
            cd = _parse_yyyy_mm_dd(call_date)
            if ed and cd:
                gap = abs((cd - ed).days)
                ev.call_gap_days = int(gap)
                if warn_gap_days and gap > int(warn_gap_days):
                    print(
                        f"[WARN] large call gap: {ev.ticker} earnings_date={ev.earnings_date} call_date={call_date} gap_days={gap}",
                        flush=True,
                    )
        except Exception:
            ev.call_gap_days = None


# -------------------------
# Robust Q&A split
# -------------------------

_QA_HEADINGS = (
    "question-and-answer",
    "question and answer",
    "questions and answers",
    "questions & answers",
    "q&a",
    "q & a",
    "q and a",
)

_QA_CUES = (
    "question-and-answer session",
    "question and answer session",
    "questions and answers session",
    "q&a session",
    "q & a session",
    "operator instructions",
    "[operator instructions]",
    "our first question",
    "your first question",
    "first question",
    "we'll go first to",
    "we will go first to",
    "we\u2019ll go first to",
    "we\u2019ll go first",
    "we\u2019ll now open",
    "we will now open",
    "we will now begin",
    "we will begin the question",
)


def _speaker_of(line: str) -> Optional[str]:
    m = SPEAKER_RE.match(line)
    if not m:
        return None
    return (m.group(1) or "").strip()


def _strip_speaker(line: str) -> str:
    m = SPEAKER_RE.match(line)
    if not m:
        return line.strip()
    return (m.group(2) or "").strip()


def _drop_leading_meta_lines(lines: List[str]) -> List[str]:
    """
    Remove leading metadata lines that are just \"Name - Firm:\" headers with no content.
    These sometimes precede the true start of the transcript (e.g., analyst roster).
    """
    i = 0
    while i < len(lines):
        ln = (lines[i] or "").strip()
        if _is_header_only(ln):
            i += 1
            continue
        break
    return lines[i:]


def find_qa_start(lines_non_empty: List[str]) -> Tuple[Optional[int], str, Optional[str]]:
    """
    Decide the first non-empty line index (0-based) that begins the Q&A section.

    Returns (idx, marker, anchor_name).

    Rules:
      - lines 0..1 are ALWAYS ineligible
      - allow pure heading lines containing Q&A headings
      - allow candidate speaker lines:
          * starts with 'Operator:'
          * OR starts with '<NAME>:' where NAME matches the speaker of line 0
        and a cue appears on that line or the immediately preceding line.
    """
    if not lines_non_empty:
        return None, "", None

    # anchor_name: the speaker of the first line (if not Operator)
    first_speaker = _speaker_of(lines_non_empty[0] or "")
    anchor_name: Optional[str] = None
    if first_speaker and first_speaker.lower() != "operator":
        anchor_name = first_speaker

    # Hard marker: first line starting with "A - "
    for i, ln in enumerate(lines_non_empty):
        if (ln or "").strip().lower().startswith("a - "):
            return i, "a_dash_marker", anchor_name

    start_i = 2  # line 3 (1-indexed)

    for i in range(start_i, len(lines_non_empty)):
        cur = (lines_non_empty[i] or "").strip()
        if not cur:
            continue

        low = cur.lower()
        prev = (lines_non_empty[i - 1] or "").strip() if i > 0 else ""
        prev_low = prev.lower()

        # heading-only triggers
        if any(h in low for h in _QA_HEADINGS):
            return i, "qna_heading", anchor_name

        # explicit question-line triggers
        if low.startswith("question:") or re.match(r"^q[\.:]\s*", low):
            return i, "question_line", anchor_name

        # candidate speaker triggers
        is_operator = low.startswith("operator:")
        is_anchor = bool(anchor_name) and low.startswith(anchor_name.lower() + ":")

        if is_operator or is_anchor:
            if any(cue in low for cue in _QA_CUES):
                return i, "speaker_line_contains_cue", anchor_name
            if any(cue in prev_low for cue in _QA_CUES):
                return i, "prev_line_contains_cue", anchor_name
    # Fallback heuristics when nothing matched above
    fallback_pats = [
        (r"\bfirst question\b", "fallback_first_question"),
        (r"\bfirst .{0,20}? question\b", "fallback_first_any_question"),
        (r"\binvestor question\b", "fallback_investor_question"),
    ]
    for pat, marker in fallback_pats:
        for i, ln in enumerate(lines_non_empty):
            if re.search(pat, ln, flags=re.IGNORECASE):
                return i, marker, anchor_name

    return None, "", anchor_name


def transcript_speaker_turns(transcript_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse transcript into speaker turns, then label each turn as prepared vs qa.

    Returns:
      turns: [{pos, section, qa_marker, speaker, text}]
      info:  {qa_start_pos, qa_marker, anchor_name, n_lines}

    Notes:
      - Uses non-empty lines only for the Q&A start detector.
      - A turn's section is decided by the position of its first contributing line.
    """
    raw_lines = transcript_text.splitlines()
    non_empty: List[str] = [ln.strip() for ln in raw_lines if (ln or "").strip()]
    non_empty = _drop_leading_meta_lines(non_empty)

    qa_start, qa_marker, anchor_name = find_qa_start(non_empty)

    turns: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for pos, line in enumerate(non_empty):
        m = SPEAKER_RE.match(line)
        if m:
            speaker = (m.group(1) or "").strip()
            content = (m.group(2) or "").strip()
            cur = {
                "pos": pos,
                "speaker": speaker,
                "text": content,
            }
            turns.append(cur)
        else:
            # continuation
            if cur is None:
                cur = {"pos": pos, "speaker": "", "text": line.strip()}
                turns.append(cur)
            else:
                cur["text"] = (cur.get("text", "") + " " + line.strip()).strip()

    # assign section
    for t in turns:
        pos = int(t.get("pos", 0))
        sec = "prepared"
        if qa_start is not None and pos >= int(qa_start):
            sec = "qa"
        t["section"] = sec
        t["qa_marker"] = qa_marker if (sec == "qa" and qa_start is not None and pos == int(qa_start)) else ""

    info = {
        "qa_start_pos": qa_start,
        "qa_marker": qa_marker,
        "anchor_name": anchor_name,
        "n_lines": len(non_empty),
    }
    return turns, info


# -------------------------
# Build transcript units per event
# -------------------------

def build_transcript_units(
    events: List[EventWindow],
    data_dir: Path,
    min_text_len: int,
) -> Tuple[pd.DataFrame, int, Dict[str, int], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    missing = 0
    section_totals = {"prepared": 0, "qa": 0}
    qa_marker_counts: Dict[str, int] = {}
    warn_path = Path("data") / "_derived" / "logs" / "transcript_warnings.log"
    _ensure_dir(warn_path.parent)

    for ev in events:
        tx_root = data_dir / ev.ticker / "transcripts"

        folder_date = None
        if ev.call_date and (tx_root / ev.call_date / "transcript.txt").exists():
            folder_date = ev.call_date
        elif (tx_root / ev.earnings_date / "transcript.txt").exists():
            folder_date = ev.earnings_date

        if not folder_date:
            missing += 1
            _log_warn(warn_path, f"{ev.ticker} {ev.earnings_date}: missing transcript folder")
            continue

        tpath = tx_root / folder_date / "transcript.txt"
        try:
            text = tpath.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            missing += 1
            _log_warn(warn_path, f"{ev.ticker} {ev.earnings_date}: failed to read transcript ({e})")
            continue

        turns, info = transcript_speaker_turns(text)

        qa_start = info.get("qa_start_pos")
        if qa_start is None:
            _log_warn(
                warn_path,
                f"{ev.ticker} {ev.earnings_date}: QA start not found; please inspect transcript at {tpath}",
            )
        elif qa_start <= 0:
            _log_warn(
                warn_path,
                f"{ev.ticker} {ev.earnings_date}: QA start at pos {qa_start}; expected prepared then QA",
            )

        for tr in turns:
            sec = str(tr.get("section", "prepared"))
            ttxt = str(tr.get("text", "") or "").strip()
            if len(ttxt) < int(min_text_len):
                continue

            section_totals[sec] = section_totals.get(sec, 0) + 1
            marker = str(tr.get("qa_marker", "") or "")
            if marker:
                qa_marker_counts[marker] = qa_marker_counts.get(marker, 0) + 1

            rows.append(
                {
                    "ticker": ev.ticker,
                    "earnings_date": ev.earnings_date,
                    "day0_date": ev.day0_date,
                    "day0_dt": ev.day0_dt or "",
                    "call_date": ev.call_date or "",
                    "call_dt": ev.call_dt or "",
                    "call_gap_days": "" if ev.call_gap_days is None else int(ev.call_gap_days),
                    "transcript_folder_date": folder_date,
                    "section": sec,
                    "qa_marker": marker,
                    "speaker": str(tr.get("speaker", "") or ""),
                    "text": ttxt,
                }
            )

    df = pd.DataFrame(rows)
    # After full pass, check global presence per event
    if "ticker" in df.columns and "earnings_date" in df.columns and "section" in df.columns:
        for ev in events:
            ev_rows = df[(df["ticker"] == ev.ticker) & (df["earnings_date"] == ev.earnings_date)]
            has_pre = (ev_rows["section"] == "prepared").any()
            has_qa = (ev_rows["section"] == "qa").any()
            if not (has_pre and has_qa):
                _log_warn(
                    warn_path,
                    f"{ev.ticker} {ev.earnings_date}: sections prepared={has_pre} qa={has_qa}; verify split logic",
                )
    else:
        for ev in events:
            _log_warn(
                warn_path,
                f"{ev.ticker} {ev.earnings_date}: no transcript rows captured; verify transcript and splitter",
            )
    return df, missing, section_totals, qa_marker_counts


# -------------------------
# Build news units per event (call-centered)
# -------------------------

def _call_anchor_dt(ev: EventWindow) -> Optional[datetime]:
    """Best-effort ET datetime for splitting pre/post."""
    if ev.call_dt:
        try:
            return to_et(parse_dt_any(ev.call_dt, assume_tz=ET))
        except Exception:
            pass

    if ev.call_date:
        dd = _parse_yyyy_mm_dd(ev.call_date)
        if dd:
            return datetime(dd.year, dd.month, dd.day, 12, 0, tzinfo=ET)

    if ev.day0_dt:
        try:
            return to_et(parse_dt_any(ev.day0_dt, assume_tz=ET))
        except Exception:
            return None

    dd = _parse_yyyy_mm_dd(ev.earnings_date)
    if dd:
        return datetime(dd.year, dd.month, dd.day, 12, 0, tzinfo=ET)
    return None


def _load_news_for_date(news_root: Path, folder_date: str) -> pd.DataFrame:
    ddir = news_root / folder_date
    if not ddir.exists():
        return pd.DataFrame()

    # Prefer raw JSON because it always includes richer fields like publisher.
    raw_fp = ddir / "stock_news.raw.json"
    if raw_fp.exists():
        try:
            obj = json.loads(raw_fp.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return pd.DataFrame(obj)
        except Exception:
            pass

    # prefer exact filename
    fp = ddir / "stock_news.csv"
    if fp.exists():
        return _safe_read_csv(fp)

    # fallback to any stock_news*.csv
    matches = sorted(ddir.glob("stock_news*.csv"))
    if matches:
        return _safe_read_csv(matches[0])

    return pd.DataFrame()


def build_news_units(
    events: List[EventWindow],
    data_dir: Path,
    min_text_len: int,
) -> Tuple[pd.DataFrame, int, Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    missing = 0
    phase_counts = {"pre": 0, "post": 0}

    for ev in events:
        news_root = data_dir / ev.ticker / "news"

        folder_date = None
        if ev.call_date and (news_root / ev.call_date).exists():
            folder_date = ev.call_date
        elif (news_root / ev.earnings_date).exists():
            folder_date = ev.earnings_date

        if not folder_date:
            missing += 1
            continue

        df = _load_news_for_date(news_root, folder_date)
        if df.empty:
            continue

        text_col = _detect_text_col(df)
        date_col = _detect_date_col(df)
        pub_col = choose_col(df, ["publisher"])
        if not text_col or not date_col:
            continue

        df = df.copy()
        df["_dt"] = df[date_col].apply(lambda x: to_et(parse_dt_any(str(x), assume_tz=ET)) if str(x).strip() else None)
        df = df.dropna(subset=["_dt"]).reset_index(drop=True)
        if df.empty:
            continue

        call_dt = _call_anchor_dt(ev)

        for _, r in df.iterrows():
            title = str(r.get("title", "") or "").strip()
            body = str(r.get(text_col, "") or "").strip()
            merged = (title + "\n" + body).strip() if title else body
            if len(merged) < int(min_text_len):
                continue

            phase = "post"
            if call_dt is not None and isinstance(r.get("_dt"), datetime):
                if r["_dt"] < call_dt:
                    phase = "pre"

            phase_counts[phase] = phase_counts.get(phase, 0) + 1

            rows.append(
                {
                    "ticker": ev.ticker,
                    "earnings_date": ev.earnings_date,
                    "day0_date": ev.day0_date,
                    "day0_dt": ev.day0_dt or "",
                    "call_date": ev.call_date or "",
                    "call_dt": ev.call_dt or "",
                    "call_gap_days": "" if ev.call_gap_days is None else int(ev.call_gap_days),
                    "news_folder_date": folder_date,
                    "phase": phase,
                    "publishedDate": str(r.get(date_col, "") or "").strip(),
                    "site": str(r.get("site", "") or "").strip(),
                    "publisher": str(r.get(pub_col, "") or "").strip(),
                    "url": str(r.get("url", "") or "").strip(),
                    "title": title,
                    "text": merged,
                }
            )

    out = pd.DataFrame(rows)
    return out, missing, phase_counts


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract news + transcript text units per event (call-centered).")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--min-text-len", type=int, default=20)

    ap.add_argument("--call-gap-warn-days", type=int, default=45)
    ap.add_argument(
        "--strict-count",
        action="store_true",
        help="Raise if events count != transcript folder count (instead of warning + tail-align).",
    )

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    min_text_len = int(args.min_text_len)

    for t in args.tickers:
        ticker = t.strip().upper()
        print(f"\n=== [TEXT UNITS] {ticker} ===", flush=True)

        events_csv = data_dir / ticker / "events" / "event_windows.csv"
        events = load_event_windows(events_csv, ticker)

        transcripts_dir = data_dir / ticker / "transcripts"
        attach_call_dates_ordered(
            events,
            transcripts_dir=transcripts_dir,
            warn_gap_days=int(args.call_gap_warn_days),
            strict_count=bool(args.strict_count),
        )

        df_news, missing_news_events, phase_counts = build_news_units(
            events, data_dir=data_dir, min_text_len=min_text_len
        )
        df_tx, missing_tx_events, sec_counts, qa_marker_counts = build_transcript_units(
            events, data_dir=data_dir, min_text_len=min_text_len
        )

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
        print(
            f"[NOTE] news phases: pre={phase_counts.get('pre',0)} post={phase_counts.get('post',0)}",
            flush=True,
        )

        if qa_marker_counts:
            top = sorted(qa_marker_counts.items(), key=lambda kv: kv[1], reverse=True)[0]
            print(f"[NOTE] top qa_marker: {top[0]}  count={top[1]}", flush=True)
        else:
            print("[NOTE] no qa_marker counts (no QA start detected or no transcripts)", flush=True)


if __name__ == "__main__":
    main()
