#!/usr/bin/env python3
# scripts/_text_utils.py

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z]+")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.upper() for t in TOKEN_RE.findall(text)]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # fallback: try python csv
        rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return pd.DataFrame(rows)


def safe_read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def parse_dt_any(x: Any) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, errors="coerce", utc=True).tz_convert(None)
    except Exception:
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return None


def to_ymd(ts: pd.Timestamp) -> str:
    return pd.to_datetime(ts).date().isoformat()


def align_to_business_day(date_ymd: str, roll: str = "forward") -> str:
    """
    Align a calendar date to the nearest business day.
    roll='forward' => weekend -> next Monday
    """
    d = np.datetime64(date_ymd)
    if np.is_busday(d):
        return str(d)
    # numpy.busday_offset returns datetime64[D]
    dd = np.busday_offset(d, 0, roll=roll)
    return str(dd)


def business_day_offset(day0_ymd: str, other_ymd: str) -> int:
    """
    Business-day offset of other relative to day0.
      other == day0 -> 0
      other > day0  -> +k
      other < day0  -> -k
    Uses numpy.busday_count which excludes end date.
    """
    if other_ymd == day0_ymd:
        return 0
    d0 = np.datetime64(day0_ymd)
    d1 = np.datetime64(other_ymd)
    if d1 > d0:
        return int(np.busday_count(d0, d1))
    else:
        return -int(np.busday_count(d1, d0))


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    candidates = [
        "text", "content", "summary", "article", "body", "description",
        "snippet", "fullText", "full_text"
    ]
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    candidates = ["publishedDate", "published_date", "date", "datetime", "time"]
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None


def split_transcript_sections(text: str) -> Dict[str, str]:
    """
    Very robust, no-assumptions splitter.
    Returns dict with keys among: prepared, qa, full
    """
    t = (text or "").strip()
    if not t:
        return {"full": ""}

    # common markers
    markers = [
        r"QUESTION[- ]AND[- ]ANSWER",
        r"Q\s*&\s*A",
        r"QUESTIONS\s+AND\s+ANSWERS",
    ]
    for pat in markers:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            prepared = t[: m.start()].strip()
            qa = t[m.start():].strip()
            out = {}
            if prepared:
                out["prepared"] = prepared
            if qa:
                out["qa"] = qa
            if not out:
                return {"full": t}
            return out

    # no marker found
    return {"full": t}
