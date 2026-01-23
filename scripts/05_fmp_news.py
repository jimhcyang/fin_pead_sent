#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Tuple

from _common import (
    FMPConfig,
    default_data_dir,
    ensure_dir,
    http_get_json,
    now_iso,
    parse_dt_any,
    to_et,
    write_json,
    write_jsonl,
)

# Stable endpoints are on the ROOT domain (not /api)
FMP_ROOT = "https://financialmodelingprep.com"
STABLE_NEWS_PATH = "/stable/news/stock"


def to_api_symbol(sym: str) -> str:
    return sym.replace(".", "-")


def parse_date_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fmt(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _extract_pub_dt(x: Dict[str, Any]) -> str | None:
    for k in ("publishedDate", "date", "published_date"):
        v = x.get(k)
        if v:
            return str(v)
    return None


def _key(x: Dict[str, Any]) -> Tuple[Any, Any]:
    # URL is best; fallback to (title, publishedDate)
    return (x.get("url"), x.get("title") or "", x.get("publishedDate") or x.get("date") or "")


def _et_day_of_item(x: Dict[str, Any]) -> date | None:
    s = _extract_pub_dt(x)
    if not s:
        return None
    try:
        dt_et = to_et(parse_dt_any(s))
        x["publishedDateET"] = dt_et.isoformat()
        return dt_et.date()
    except Exception:
        return None


def fetch_news_for_day_paginated(
    cfg: FMPConfig,
    symbol: str,
    target_day: str,
    pad_days: int = 1,
    page_limit: int = 100,
    max_pages: int = 200,
    sleep_s: float = 0.2,
) -> Dict[str, Any]:
    """
    Writes ONE-day output, but queries [D-pad_days, D+pad_days] to avoid UTC/ET boundary loss,
    paginates with page+limit, and filters to ET date == target_day.

    Returns dict with:
      - items (filtered, deduped)
      - stats (pages, raw counts, etc.)
      - query_window
    """
    ticker = symbol.upper()
    api_symbol = to_api_symbol(ticker)

    d0 = parse_date_ymd(target_day)
    q_from = d0 - timedelta(days=pad_days)
    q_to = d0 + timedelta(days=pad_days)

    url = f"{FMP_ROOT}{STABLE_NEWS_PATH}"

    raw_items: List[Dict[str, Any]] = []
    seen_raw_keys = set()
    pages_fetched = 0

    for page in range(max_pages):
        params = {
            "symbols": api_symbol,
            "from": fmt(q_from),
            "to": fmt(q_to),
            "limit": page_limit,
            "page": page,
            "apikey": cfg.api_key,
        }

        batch = http_get_json(url, params=params, sleep_s=sleep_s)

        if not isinstance(batch, list) or not batch:
            break

        pages_fetched += 1

        # Add only NEW raw items; if a whole page repeats, we stop.
        new_in_page = 0
        for x in batch:
            k = _key(x)
            if k in seen_raw_keys:
                continue
            seen_raw_keys.add(k)
            raw_items.append(x)
            new_in_page += 1

        # If API is ignoring 'page' and repeating page 0, new_in_page becomes 0 → break
        if new_in_page == 0:
            break

        # If this was the last page in the API sense, break
        if len(batch) < page_limit:
            break

    # Filter to target day in ET
    filtered: List[Dict[str, Any]] = []
    for x in raw_items:
        d_et = _et_day_of_item(x)
        if d_et == d0:
            filtered.append(x)

    # Dedup again after ET conversion (safe)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for x in filtered:
        k = _key(x)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(x)

    deduped.sort(key=lambda x: str(x.get("publishedDateET", x.get("publishedDate", ""))), reverse=True)

    return {
        "items": deduped,
        "stats": {
            "ticker": ticker,
            "api_symbol_used": api_symbol,
            "target_day_et": target_day,
            "pad_days": pad_days,
            "page_limit": page_limit,
            "max_pages": max_pages,
            "pages_fetched": pages_fetched,
            "raw_unique_items": len(raw_items),
            "filtered_items_et_day": len(filtered),
            "final_deduped_items": len(deduped),
            "base_url_used": FMP_ROOT,
        },
        "query_window": {"from": fmt(q_from), "to": fmt(q_to)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (this is the ET day you want to SAVE)")
    ap.add_argument("--pad-days", type=int, default=1, help="Internal query padding to handle UTC/ET boundaries")
    ap.add_argument("--page-limit", type=int, default=100)
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    cfg = FMPConfig.from_env()
    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    out_dir = data_dir / ticker / "news"
    ensure_dir(out_dir)

    res = fetch_news_for_day_paginated(
        cfg,
        symbol=ticker,
        target_day=args.date,
        pad_days=args.pad_days,
        page_limit=args.page_limit,
        max_pages=args.max_pages,
        sleep_s=args.sleep,
    )

    items = res["items"]
    stats = res["stats"]
    qwin = res["query_window"]

    out_path = out_dir / f"news_{args.date}.jsonl"
    n = write_jsonl(items, out_path)

    meta = {
        **stats,
        "records_written": n,
        "query_window_used": qwin,
        "downloaded_at_et": now_iso(),
        "source": "FMP /stable/news/stock (paginated page+limit; filtered to ET day)",
    }
    write_json(meta, out_dir / f"news_{args.date}.meta.json")

    print(f"[OK] Wrote {out_path} ({n} records)")
    print(f"[OK] Pages fetched: {stats['pages_fetched']} | raw unique: {stats['raw_unique_items']} | final: {stats['final_deduped_items']}")
    print(f"[OK] Internal query window: {qwin['from']} .. {qwin['to']}")


if __name__ == "__main__":
    main()
