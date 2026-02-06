#!/usr/bin/env python3
# scripts/31_extract_text_units.py
# Extract event-aligned text units (news + transcripts) for later scoring.
#
# Key design decisions:
# - News: each unit = one article (title+text+content merged), assigned to the *closest* earnings event
#   within that event's [dt_m5, dt_p10] trading-day window; phase=pre/post based on rel trading day vs day0.
# - Transcripts: each unit = one speaker turn, split into section={prepared, qa} using robust Q&A markers.
#   We keep "Operator" turns but tag role=operator so downstream scoring can include/exclude them easily.
#
# Output (per ticker):
#   data/{T}/events/text_units_news.csv
#   data/{T}/events/text_units_transcripts.csv
#   data/{T}/events/text_units_meta.json
#
# Optional pooled output:
#   data/_derived/text/text_units_news.csv
#   data/_derived/text/text_units_transcripts.csv
#   data/_derived/text/text_units.meta.json

from __future__ import annotations

import argparse
import csv
import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from _common import DEFAULT_20

NY_TZ = "America/New_York"


def read_tickers_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"{path} is empty.")
            ticker_col = next((c for c in reader.fieldnames if c.strip().lower() == "ticker"), None)
            if ticker_col is None:
                raise ValueError(f"{path} has no 'ticker' column.")
            out: list[str] = []
            for row in reader:
                t = (row.get(ticker_col) or "").strip()
                if t:
                    out.append(t)
            return out
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


@dataclass(frozen=True)
class EventWindow:
    ticker: str
    earnings_date: str
    day0_date: str
    dt_m5: str
    dt_p10: str


def _to_day_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert any timestamp (tz-aware or tz-naive) to a tz-naive *day* timestamp in NY."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts.normalize()


def _to_date_str(ts: pd.Timestamp) -> str:
    return _to_day_naive(ts).date().isoformat()


def load_event_windows(data_dir: Path, ticker: str) -> List[EventWindow]:
    p = data_dir / ticker / "events" / "event_windows.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run the 20s event pipeline first.")
    df = pd.read_csv(p)
    need = {"ticker", "earnings_date", "day0_date", "dt_m5", "dt_p10"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{p} missing columns: {sorted(miss)}")
    out: list[EventWindow] = []
    for _, r in df.iterrows():
        out.append(
            EventWindow(
                ticker=str(r["ticker"]).upper(),
                earnings_date=str(r["earnings_date"])[:10],
                day0_date=str(r["day0_date"])[:10],
                dt_m5=str(r["dt_m5"])[:10],
                dt_p10=str(r["dt_p10"])[:10],
            )
        )
    return out


def load_trading_days(data_dir: Path, ticker: str) -> List[pd.Timestamp]:
    p = data_dir / ticker / "prices" / "yf_ohlcv_daily.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run scripts/01_yf_prices.py first.")
    df = pd.read_csv(p)
    if "date" not in df.columns:
        raise ValueError(f"{p} missing 'date'")
    d = pd.to_datetime(df["date"], errors="coerce").dropna().sort_values().unique().tolist()
    out = [_to_day_naive(pd.Timestamp(x)) for x in d]
    out = sorted(set(out))
    return out


def rel_trading_day(trading_days: List[pd.Timestamp], day0: pd.Timestamp, d: pd.Timestamp) -> Optional[int]:
    idx = {td: i for i, td in enumerate(trading_days)}
    day0 = _to_day_naive(day0)
    d = _to_day_naive(d)
    if day0 not in idx or d not in idx:
        return None
    return int(idx[d] - idx[day0])


def _map_to_trading_day(trading_days: List[pd.Timestamp], d: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Map a calendar day to the most recent trading day <= that day.
    Assumes trading_days is sorted, tz-naive midnight timestamps.
    """
    if not trading_days:
        return None
    day = _to_day_naive(d)
    i = bisect_right(trading_days, day) - 1
    if i < 0:
        return None
    return trading_days[i]


def load_all_news(data_dir: Path, ticker: str) -> pd.DataFrame:
    base = data_dir / ticker / "news"
    if not base.exists():
        return pd.DataFrame()
    parts = []
    for sub in sorted(base.glob("*")):
        if not sub.is_dir():
            continue
        p = sub / "stock_news.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["_src_file"] = str(p)
            df["_src_row"] = range(len(df))
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0, ignore_index=True)


def _coalesce_text(row: pd.Series) -> str:
    # FMP news usually has title + text; sometimes "content" exists too
    title = str(row.get("title", "") or "").strip()
    text = str(row.get("text", "") or "").strip()
    content = str(row.get("content", "") or "").strip()
    merged = "\n".join([x for x in [title, text, content] if x])
    return merged.strip()


def _get_news_published_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    """
    FMP 'stock_news.csv' commonly contains:
      - date_et (YYYY-MM-DD)  [preferred for day alignment in ET]
      - publishedDateET (ISO with -04:00/-05:00)
      - publishedDate (often naive string)
      - date (sometimes)
    Return a pd.Timestamp (may be tz-aware) representing publication time/day.
    """
    for k in ["date_et", "publishedDateET", "publishedDate", "date"]:
        v = row.get(k, None)
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        # date_et is already a date string; treat as NY day (tz-naive)
        if k == "date_et":
            ts = pd.to_datetime(s, errors="coerce")
            if pd.isna(ts):
                continue
            return _to_day_naive(ts)
        # For other datetime strings, let pandas infer tz; then normalize to NY day
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            continue
        # If tz-aware, convert to NY before dropping tz
        if ts.tzinfo is not None:
            ts = ts.tz_convert(NY_TZ)
        return ts
    return None


def build_news_units(
    ticker: str,
    events: List[EventWindow],
    trading_days: List[pd.Timestamp],
    news_raw: pd.DataFrame,
    min_text_len: int,
) -> pd.DataFrame:
    if news_raw is None or news_raw.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "earnings_date",
                "day0_date",
                "published_date",
                "published_trading_date",
                "rel_td",
                "phase",
                "title",
                "text",
                "site",
                "url",
                "unit_id",
                "_src_file",
                "_src_row",
            ]
        )

    # Build event ranges [m5, p10] in trading-day space
    ev_ranges: list[tuple[EventWindow, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for ev in events:
        a = pd.to_datetime(ev.dt_m5, errors="coerce")
        b = pd.to_datetime(ev.dt_p10, errors="coerce")
        d0 = pd.to_datetime(ev.day0_date, errors="coerce")
        if pd.isna(a) or pd.isna(b) or pd.isna(d0):
            continue
        ev_ranges.append((ev, _to_day_naive(a), _to_day_naive(b), _to_day_naive(d0)))

    rows: list[dict] = []
    t = ticker.upper()

    for _, r in news_raw.iterrows():
        pub_ts = _get_news_published_ts(r)
        if pub_ts is None:
            continue

        merged_text = _coalesce_text(r)
        if len(merged_text) < int(min_text_len):
            continue

        # map to trading day (most recent trading day <= published day in NY)
        pub_day = _to_day_naive(pub_ts)
        pub_trade = _map_to_trading_day(trading_days, pub_day)
        if pub_trade is None:
            continue

        # assign to an event if within [m5, p10] trading day
        assigned: Optional[EventWindow] = None
        best_abs: Optional[int] = None

        for ev, a, b, d0 in ev_ranges:
            if a <= pub_trade <= b:
                rt = rel_trading_day(trading_days, d0, pub_trade)
                if rt is None:
                    continue
                aa = abs(int(rt))
                if best_abs is None or aa < best_abs:
                    best_abs = aa
                    assigned = ev

        if assigned is None:
            continue

        d0 = _to_day_naive(pd.to_datetime(assigned.day0_date, errors="coerce"))
        rt = rel_trading_day(trading_days, d0, pub_trade)
        if rt is None:
            continue

        phase = "pre" if rt < 0 else "post"  # include day0 in post
        unit_id = f"{t}|{assigned.earnings_date}|news|{r.get('_src_file','')}|{int(r.get('_src_row',0))}"

        rows.append(
            {
                "ticker": t,
                "earnings_date": assigned.earnings_date,
                "day0_date": assigned.day0_date,
                "published_date": _to_date_str(pub_day),
                "published_trading_date": _to_date_str(pub_trade),
                "rel_td": int(rt),
                "phase": phase,
                "title": str(r.get("title", "") or ""),
                "text": merged_text,
                "site": str(r.get("site", "") or ""),
                "url": str(r.get("url", "") or ""),
                "unit_id": unit_id,
                "_src_file": str(r.get("_src_file", "") or ""),
                "_src_row": int(r.get("_src_row", 0) or 0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # de-dup on url+published_trading_date within event
    if out["url"].astype(str).str.strip().ne("").any():
        out = out.sort_values(["ticker", "earnings_date", "published_trading_date"]).drop_duplicates(
            subset=["ticker", "earnings_date", "url", "published_trading_date"],
            keep="first",
        )

    return out.reset_index(drop=True)


def read_transcript_content(folder: Path) -> str:
    # Prefer transcript.txt; fall back to transcript.json if needed
    txt = folder / "transcript.txt"
    if txt.exists():
        return txt.read_text(encoding="utf-8", errors="ignore")

    js = folder / "transcript.json"
    if js.exists():
        try:
            obj = json.loads(js.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, dict):
                if isinstance(obj.get("content", None), str):
                    return obj["content"]
                if isinstance(obj.get("transcript", None), str):
                    return obj["transcript"]
        except Exception:
            return ""
    return ""


_QA_MARKERS: list[tuple[str, str]] = [
    (r"\bQUESTION[- ]AND[- ]ANSWER SESSION\b", "qna_heading"),
    (r"\bQUESTIONS\s+AND\s+ANSWERS\b", "questions_and_answers"),
    (r"\bQ&A\b", "q_and_a"),
    # FMP-style operator handoff to Q&A (works for many transcripts)
    (r"(?im)^\s*Operator:\s*(?:we['’]ll|we will|let['’]s)\s+(?:now\s+)?(?:take|begin)\s+(?:our\s+)?(?:first|next)\s+question\b", "operator_first_question"),
    (r"(?im)^\s*Operator:.*\b(first|next)\s+question\b", "operator_contains_first_question"),
    (r"(?im)^\s*Operator:.*\bquestions\b", "operator_contains_questions"),
]


def split_prepared_vs_qa(text: str) -> tuple[str, str, dict]:
    """
    Split a transcript into prepared remarks vs Q&A.

    Returns (prepared_text, qa_text, info_dict).
    If no marker found, returns (full_text, "", {marker:None}).

    NOTE: This is heuristic by necessity: the raw transcript text does not always include
    explicit structural tags. We bias toward finding the *earliest* plausible Q&A start.
    """
    if not text:
        return "", "", {"marker": None, "pos": None}

    best = None  # (start, end, name)
    for pat, name in _QA_MARKERS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        if best is None or m.start() < best[0]:
            best = (m.start(), m.end(), name)

    if best is None:
        return text, "", {"marker": None, "pos": None}

    return text[: best[0]], text[best[0] :], {"marker": best[2], "pos": int(best[0])}


def parse_speaker_turns(section_text: str) -> list[dict]:
    """
    Speaker-turn parser supporting two common formats:
      A) "Speaker: words..." (colon format)
      B) "Speaker - Role" or "Speaker – Role" (dash format)

    Returns list of dicts:
      {turn_index, speaker, role_hint, text}
    """
    lines = (section_text or "").splitlines()
    turns: list[dict] = []
    cur_speaker = ""
    cur_role_hint = ""
    buf: list[str] = []
    idx = 0

    colon_pat = re.compile(r"^([A-Z][A-Za-z0-9 .,'&/()\-]{1,80}):\s*(.*)$")
    dash_pat = re.compile(r"^([A-Z][A-Za-z0-9 .,'&/()\-]{1,80})\s*[-–]\s*(.*)$")

    def flush():
        nonlocal idx, buf, cur_speaker, cur_role_hint
        txt = "\n".join([x for x in buf if x.strip()]).strip()
        if txt:
            turns.append(
                {
                    "turn_index": idx,
                    "speaker": cur_speaker.strip(),
                    "role_hint": cur_role_hint.strip(),
                    "text": txt,
                }
            )
            idx += 1
        buf = []

    for ln in lines:
        s = ln.strip()
        if not s:
            continue

        m = colon_pat.match(s)
        if m:
            flush()
            cur_speaker = m.group(1).strip()
            cur_role_hint = ""  # colon format usually lacks explicit role
            rest = (m.group(2) or "").strip()
            if rest:
                buf.append(rest)
            continue

        m2 = dash_pat.match(s)
        if m2:
            flush()
            cur_speaker = m2.group(1).strip()
            cur_role_hint = (m2.group(2) or "").strip()
            continue

        buf.append(s)

    flush()
    return turns


def speaker_role(speaker: str, role_hint: str, section: str) -> str:
    sp = (speaker or "").strip().lower()
    rh = (role_hint or "").strip().lower()
    if sp == "operator" or "operator" in sp:
        return "operator"
    # if the transcript explicitly includes roles in the dash format
    if "analyst" in rh or "investor" in rh:
        return "analyst"
    if any(k in rh for k in ["chief", "ceo", "cfo", "president", "vp", "vice president", "director"]):
        return "executive"
    # weak heuristic: in prepared section, non-operator is likely executive
    if (section or "").lower() == "prepared" and sp:
        return "executive_or_company"
    return "unknown"


def build_transcript_units(
    data_dir: Path,
    ticker: str,
    events: List[EventWindow],
    min_text_len: int,
) -> pd.DataFrame:
    t = ticker.upper()
    base = data_dir / t / "transcripts"
    if not base.exists():
        return pd.DataFrame(
            columns=[
                "ticker",
                "earnings_date",
                "day0_date",
                "section",
                "speaker",
                "role",
                "role_hint",
                "turn_index",
                "text",
                "unit_id",
                "qa_marker",
            ]
        )

    rows: list[dict] = []
    for ev in events:
        folder = base / ev.earnings_date
        if not folder.exists():
            continue

        content = read_transcript_content(folder)
        if not content:
            continue

        prepared, qa, info = split_prepared_vs_qa(content)
        marker = info.get("marker", None)

        for section_name, sec_text in [("prepared", prepared), ("qa", qa)]:
            if not sec_text or not sec_text.strip():
                continue
            turns = parse_speaker_turns(sec_text)
            for tr in turns:
                txt = str(tr.get("text", "") or "").strip()
                if len(txt) < int(min_text_len):
                    continue
                spk = str(tr.get("speaker", "") or "").strip()
                rh = str(tr.get("role_hint", "") or "").strip()
                rid = int(tr.get("turn_index", 0) or 0)

                unit_id = f"{t}|{ev.earnings_date}|transcript|{section_name}|{rid}"
                rows.append(
                    {
                        "ticker": t,
                        "earnings_date": ev.earnings_date,
                        "day0_date": ev.day0_date,
                        "section": section_name,
                        "speaker": spk,
                        "role_hint": rh,
                        "role": speaker_role(spk, rh, section_name),
                        "turn_index": rid,
                        "text": txt,
                        "unit_id": unit_id,
                        "qa_marker": marker,
                    }
                )

    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract event-aligned text units (news + transcripts).")

    ap.add_argument("--ticker", default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--tickers-file", default=None)
    ap.add_argument("--max-tickers", type=int, default=None)

    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--min-text-len", type=int, default=20)
    ap.add_argument("--also-pooled", action="store_true")

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

    pooled_news: list[pd.DataFrame] = []
    pooled_tr: list[pd.DataFrame] = []
    pooled_meta: dict = {"tickers": tickers, "per_ticker": {}}

    for t in tickers:
        t = t.upper()
        print(f"\n=== [TEXT UNITS] {t} ===", flush=True)

        events = load_event_windows(data_dir, t)
        trading_days = load_trading_days(data_dir, t)

        news_raw = load_all_news(data_dir, t)
        news_units = build_news_units(t, events, trading_days, news_raw, min_text_len=int(args.min_text_len))
        tr_units = build_transcript_units(data_dir, t, events, min_text_len=int(args.min_text_len))

        out_dir = data_dir / t / "events"
        out_dir.mkdir(parents=True, exist_ok=True)

        news_path = out_dir / "text_units_news.csv"
        tr_path = out_dir / "text_units_transcripts.csv"
        meta_path = out_dir / "text_units_meta.json"

        news_units.to_csv(news_path, index=False)
        tr_units.to_csv(tr_path, index=False)

        # notes on missingness
        ev_set = {ev.earnings_date for ev in events}
        news_ev = set(news_units["earnings_date"].unique().tolist()) if not news_units.empty else set()
        tr_ev = set(tr_units["earnings_date"].unique().tolist()) if not tr_units.empty else set()

        mt = {
            "ticker": t,
            "events_total": len(events),
            "missing_news_events": int(len(ev_set - news_ev)),
            "missing_transcript_events": int(len(ev_set - tr_ev)),
            "news_units": int(len(news_units)),
            "transcript_units": int(len(tr_units)),
            "transcript_units_prepared": int((tr_units["section"] == "prepared").sum()) if not tr_units.empty else 0,
            "transcript_units_qa": int((tr_units["section"] == "qa").sum()) if not tr_units.empty else 0,
            "qa_marker_counts": tr_units["qa_marker"].value_counts(dropna=False).to_dict() if not tr_units.empty else {},
        }
        pooled_meta["per_ticker"][t] = mt
        meta_path.write_text(json.dumps(mt, indent=2), encoding="utf-8")

        print(f"[OK] wrote news units       -> {news_path}  rows={len(news_units):,}")
        print(f"[OK] wrote transcript units -> {tr_path}  rows={len(tr_units):,}")
        print(
            f"[NOTE] events_total={mt['events_total']} missing_news_events={mt['missing_news_events']} "
            f"missing_transcript_events={mt['missing_transcript_events']}"
        )
        if not tr_units.empty:
            print(f"[NOTE] transcript sections: prepared={mt['transcript_units_prepared']:,} qa={mt['transcript_units_qa']:,}")
            if mt["qa_marker_counts"]:
                # show the top marker
                top_marker = sorted(mt["qa_marker_counts"].items(), key=lambda x: -x[1])[0]
                print(f"[NOTE] top qa_marker: {top_marker[0]}  count={top_marker[1]}")

        if args.also_pooled:
            if not news_units.empty:
                pooled_news.append(news_units)
            if not tr_units.empty:
                pooled_tr.append(tr_units)

    if args.also_pooled:
        pooled_dir = data_dir / "_derived" / "text"
        pooled_dir.mkdir(parents=True, exist_ok=True)

        if pooled_news:
            pd.concat(pooled_news, ignore_index=True).to_csv(pooled_dir / "text_units_news.csv", index=False)
        if pooled_tr:
            pd.concat(pooled_tr, ignore_index=True).to_csv(pooled_dir / "text_units_transcripts.csv", index=False)

        (pooled_dir / "text_units.meta.json").write_text(json.dumps(pooled_meta, indent=2), encoding="utf-8")
        print(f"[OK] wrote pooled -> {pooled_dir}")


if __name__ == "__main__":
    main()
