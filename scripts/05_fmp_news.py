#!/usr/bin/env python3
from __future__ import annotations

"""scripts/05_fmp_news.py

Download stock news from Financial Modeling Prep (FMP) "stable" endpoint with pagination.

Endpoint:
  https://financialmodelingprep.com/stable/news/stock

Mode A: earnings/event-window (recommended)
  Pass an earnings date (day 0). The script downloads news from:
    day -PRE_BDAYS (weekdays-only) to day +POST_BDAYS (weekdays-only), inclusive.

  Example (day0=2024-01-10, pre=5, post=10): from=2024-01-03 to=2024-01-24

  Output directory:
    data/{TICKER}/news/{EARNINGS_DATE}/

Mode B: explicit date range
  Provide --start and --end (YYYY-MM-DD). Output directory:
    data/{TICKER}/news/

Notes:
  - "Business days" here means weekdays only (Mon–Fri). US market holidays are NOT excluded.
  - We fetch with a small pad around the window then filter back to the exact ET date window.
"""

import argparse
import csv
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

from _common import ensure_dir, parse_dt_any, to_et, http_get_json

FMP_ROOT = "https://financialmodelingprep.com"
STOCK_NEWS_ENDPOINT = f"{FMP_ROOT}/stable/news/stock"

ET = ZoneInfo("America/New_York")


def _parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def shift_weekdays(d: date, n: int) -> date:
    """Shift by n weekdays (Mon–Fri). Holidays are NOT excluded."""
    if n == 0:
        return d
    step = 1 if n > 0 else -1
    cur = d
    for _ in range(abs(int(n))):
        cur = cur + timedelta(days=step)
        while cur.weekday() >= 5:  # 5=Sat, 6=Sun
            cur = cur + timedelta(days=step)
    return cur


def _daterange_chunks(start_d: date, end_d: date, chunk_days: int) -> List[Tuple[date, date]]:
    """Inclusive chunks covering [start_d, end_d]. chunk_days<=0 => single chunk."""
    if chunk_days <= 0:
        return [(start_d, end_d)]

    total_days = (end_d - start_d).days + 1
    if chunk_days >= total_days:
        return [(start_d, end_d)]

    out: List[Tuple[date, date]] = []
    cur = start_d
    while cur <= end_d:
        nxt = min(end_d, cur + timedelta(days=chunk_days - 1))
        out.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return out


def _canonical_key(item: Dict[str, Any]) -> str:
    """Stable-ish key for de-duplication."""
    url = (item.get("url") or item.get("link") or "").strip().lower()
    if url:
        return url

    title = (item.get("title") or "").strip().lower()
    pd = (item.get("publishedDate") or item.get("date") or "").strip().lower()
    site = (item.get("site") or "").strip().lower()
    return f"{title}::{pd}::{site}"


def _normalize_dates(item: Dict[str, Any]) -> None:
    """Add publishedDateET + date_et (YYYY-MM-DD) in-place if possible."""
    raw = (item.get("publishedDate") or item.get("date") or "").strip()
    if not raw:
        return

    # FMP sometimes returns naive timestamps. Assume ET for naive.
    try:
        dt = parse_dt_any(raw, assume_tz=ET)
    except Exception:
        return

    dt_et = to_et(dt)
    item["publishedDateET"] = dt_et.isoformat()
    item["date_et"] = dt_et.date().isoformat()


def _in_date_window(item: Dict[str, Any], start_d: date, end_d: date) -> bool:
    d = (item.get("date_et") or "").strip()
    if not d:
        return False
    try:
        dd = _parse_yyyy_mm_dd(d)
    except Exception:
        return False
    return start_d <= dd <= end_d


def fetch_stock_news_paginated(
    api_key: str,
    api_symbol: str,
    from_d: date,
    to_d: date,
    *,
    page_limit: int,
    max_pages: int,
    sleep_s: float,
) -> List[Dict[str, Any]]:
    """Fetch stock news from [from_d,to_d] (inclusive) with pagination."""
    out: List[Dict[str, Any]] = []

    # Conservative clamps (FMP plan tiers may be stricter).
    page_limit = max(1, min(int(page_limit), 250))
    max_pages = max(1, min(int(max_pages), 100))

    last_page_fp: Optional[str] = None
    repeated_pages = 0

    for page in range(max_pages):
        params = {
            "symbols": api_symbol,
            "from": from_d.isoformat(),
            "to": to_d.isoformat(),
            "page": page,
            "limit": int(page_limit),
            "apikey": api_key,
        }

        data = http_get_json(STOCK_NEWS_ENDPOINT, params=params, sleep_s=sleep_s)

        # FMP sometimes returns dict error payloads.
        if isinstance(data, dict):
            if any(str(k).lower().startswith("error") for k in data.keys()):
                raise RuntimeError(f"FMP error payload: {data}")
            raise RuntimeError(f"Unexpected dict payload for stock news: {list(data.keys())[:10]}")

        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected payload type for stock news: {type(data)}")

        if not data:
            break

        # Detect non-paginating behavior (same payload every page)
        fp = _canonical_key(data[0]) + "::" + str(len(data))
        if last_page_fp is not None and fp == last_page_fp:
            repeated_pages += 1
            if repeated_pages >= 2:
                # API is likely ignoring `page`; stop to avoid infinite loop.
                break
        else:
            repeated_pages = 0
        last_page_fp = fp

        out.extend(data)

        # If we got fewer than page_limit, likely last page.
        if len(data) < page_limit:
            break

    return out


def write_csv(records: List[Dict[str, Any]], path: Path) -> int:
    ensure_dir(path.parent)
    cols = [
        "ticker",
        "event_date",
        "window_start",
        "window_end",
        "symbol",
        "symbols",
        "publishedDate",
        "publishedDateET",
        "date_et",
        "title",
        "text",
        "site",
        "url",
        "image",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            row = {c: r.get(c, "") for c in cols}
            w.writerow(row)

    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download FMP STABLE stock news for a ticker over a date range.")
    ap.add_argument("--ticker", required=True)

    # Mode A: earnings/event window
    ap.add_argument(
        "--earnings-date",
        default=None,
        help="Event date (YYYY-MM-DD). If set, downloads [-pre-bdays,+post-bdays] weekdays-only window.",
    )
    ap.add_argument("--pre-bdays", type=int, default=5, help="Weekday offset before event date (default: 5).")
    ap.add_argument("--post-bdays", type=int, default=10, help="Weekday offset after event date (default: 10).")

    # Mode B: explicit date range
    ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD (explicit-range mode)")
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD (explicit-range mode)")
    ap.add_argument("--date", default=None, help="Optional: single date YYYY-MM-DD (explicit-range; overrides --start/--end)")

    ap.add_argument("--data-dir", default=None, help="Default: repo_root/data")
    ap.add_argument("--api-key", default=None, help="FMP API key. If omitted, uses env var FMP_API_KEY.")

    ap.add_argument("--page-limit", type=int, default=100)
    ap.add_argument("--max-pages", type=int, default=100)
    ap.add_argument("--chunk-days", type=int, default=90)
    ap.add_argument("--pad-days", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.2)

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    args = ap.parse_args()

    ticker = args.ticker.strip().upper()
    api_symbol = ticker.replace(".", "-")

    # API key: prefer explicit flag, else env var.
    api_key = (args.api_key or os.environ.get("FMP_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("Missing FMP API key. Provide --api-key or set FMP_API_KEY in the environment.")

    # Choose mode + compute window.
    event_date: Optional[str] = None
    if args.earnings_date:
        event_date = args.earnings_date.strip()
        day0 = _parse_yyyy_mm_dd(event_date)
        start_d = shift_weekdays(day0, -int(args.pre_bdays))
        end_d = shift_weekdays(day0, int(args.post_bdays))
    else:
        # Explicit-range mode
        if args.date:
            start_d = end_d = _parse_yyyy_mm_dd(args.date)
        else:
            if not args.start or not args.end:
                raise ValueError("Provide --earnings-date OR (--)start and --end (or --date).")
            start_d = _parse_yyyy_mm_dd(args.start)
            end_d = _parse_yyyy_mm_dd(args.end)

    if start_d > end_d:
        raise ValueError(f"start > end: {start_d} > {end_d}")

    base_dir = Path(args.data_dir) if args.data_dir else Path("data")
    out_dir = base_dir / ticker / "news"
    if event_date:
        out_dir = out_dir / event_date
    ensure_dir(out_dir)

    # Filename stem
    if event_date:
        stem = "stock_news"
    else:
        stem = f"stock_news_{start_d.isoformat()}_{end_d.isoformat()}"

    raw_path = out_dir / f"{stem}.raw.json"
    jsonl_path = out_dir / f"{stem}.jsonl"
    csv_path = out_dir / f"{stem}.csv"
    meta_path = out_dir / f"{stem}.meta.json"

    if (raw_path.exists() and csv_path.exists() and meta_path.exists()) and not args.overwrite:
        print(f"[SKIP] {ticker}: news already exists -> {out_dir}")
        return

    chunks = _daterange_chunks(start_d, end_d, int(args.chunk_days))
    pad = max(0, int(args.pad_days))

    seen: set[str] = set()
    kept: List[Dict[str, Any]] = []

    total_fetched = 0
    total_dupes = 0

    for (c0, c1) in chunks:
        req_from = c0 - timedelta(days=pad)
        req_to = c1 + timedelta(days=pad)

        batch = fetch_stock_news_paginated(
            api_key,
            api_symbol,
            req_from,
            req_to,
            page_limit=int(args.page_limit),
            max_pages=int(args.max_pages),
            sleep_s=float(args.sleep),
        )
        total_fetched += len(batch)

        for item in batch:
            if not isinstance(item, dict):
                continue

            item = dict(item)  # copy
            item["ticker"] = ticker
            item["event_date"] = event_date or ""
            item["window_start"] = start_d.isoformat()
            item["window_end"] = end_d.isoformat()

            _normalize_dates(item)

            # Filter to exact requested window
            if not _in_date_window(item, start_d, end_d):
                continue

            k = _canonical_key(item)
            if k in seen:
                total_dupes += 1
                continue
            seen.add(k)

            kept.append(item)

    # Sort by ET timestamp if available
    def _sort_key(x: Dict[str, Any]) -> str:
        return (x.get("publishedDateET") or x.get("publishedDate") or "")

    kept.sort(key=_sort_key)

    # Write files
    raw_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")

    # JSONL
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_csv = write_csv(kept, csv_path)

    meta: Dict[str, Any] = {
        "ticker": ticker,
        "api_symbol": api_symbol,
        "endpoint": STOCK_NEWS_ENDPOINT,
        "mode": "earnings_window" if event_date else "explicit_range",
        "earnings_date": event_date,
        "pre_bdays": int(args.pre_bdays) if event_date else None,
        "post_bdays": int(args.post_bdays) if event_date else None,
        "start": start_d.isoformat(),
        "end": end_d.isoformat(),
        "chunks": [{"start": a.isoformat(), "end": b.isoformat()} for (a, b) in chunks],
        "page_limit": int(args.page_limit),
        "max_pages": int(args.max_pages),
        "chunk_days": int(args.chunk_days),
        "pad_days": pad,
        "sleep": float(args.sleep),
        "fetched_total": int(total_fetched),
        "kept_total": int(len(kept)),
        "dropped_dupes": int(total_dupes),
        "min_publishedDateET": (kept[0].get("publishedDateET") if kept else None),
        "max_publishedDateET": (kept[-1].get("publishedDateET") if kept else None),
        "outputs": {
            "raw_json": str(raw_path),
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
        },
        "generated_at": datetime.now(ET).isoformat(),
        "notes": "Dates are filtered using America/New_York (ET) day boundaries. Business days are Mon–Fri only; US market holidays are not excluded.",
    }

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] {ticker}: news {start_d}..{end_d} kept={len(kept)} (fetched={total_fetched}, dupes={total_dupes})")
    print(f"      -> {csv_path}")


if __name__ == "__main__":
    main()
