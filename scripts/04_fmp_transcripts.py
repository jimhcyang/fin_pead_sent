#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _common import (
    FMPConfig,
    default_data_dir,
    ensure_dir,
    http_get_json,
    now_iso,
    parse_dt_any,
    to_et,
    write_json,
)

DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# FMP commonly: /v3/earning_call_transcript/{symbol}?quarter=3&year=2025&apikey=...
def fetch_one(cfg: FMPConfig, ticker: str, year: int, quarter: int, sleep_s: float) -> List[Dict[str, Any]]:
    url = f"{cfg.base_url}/v3/earning_call_transcript/{ticker}"
    params = {"year": year, "quarter": quarter, "apikey": cfg.api_key}
    data = http_get_json(url, params=params, sleep_s=sleep_s)
    if isinstance(data, dict):
        # sometimes a single record
        return [data]
    if isinstance(data, list):
        return data
    return []


def in_window(call_date: str, start: str, end: str) -> bool:
    return (call_date >= start) and (call_date <= end)


def _call_date_key(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    # FMP sometimes returns "YYYY-MM-DD HH:MM:SS" etc
    return s.split(" ")[0].strip() if " " in s else s


def _list_date_dirs(transcripts_dir: Path) -> List[Path]:
    if not transcripts_dir.exists():
        return []
    out: List[Path] = []
    for child in transcripts_dir.iterdir():
        if child.is_dir() and DATE_DIR_RE.match(child.name):
            out.append(child)
    return sorted(out, key=lambda p: p.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--expected", type=int, default=20, help="Keep only the last N call dates (default 20).")
    ap.add_argument("--fiscal-year-min", type=int, default=2020)
    ap.add_argument("--fiscal-year-max", type=int, default=2026)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument(
        "--no-prune-extra",
        action="store_true",
        help="Do not delete transcript date folders outside the last expected set.",
    )
    args = ap.parse_args()

    cfg = FMPConfig.from_env()
    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    expected = int(args.expected)

    out_base = data_dir / ticker / "transcripts"
    ensure_dir(out_base)

    # Collect first, then write only last N (prevents 21)
    # Keep the "best" record per date (prefer longer content).
    by_date: Dict[str, Tuple[Dict[str, Any], str, int, int]] = {}  # call_date -> (record, content, year, quarter)

    for year in range(args.fiscal_year_min, args.fiscal_year_max + 1):
        for q in [1, 2, 3, 4]:
            try:
                recs = fetch_one(cfg, ticker, year, q, args.sleep)
            except Exception:
                continue

            for r in recs:
                d_raw = str(r.get("date", "")).strip()
                content = r.get("content", None) or r.get("transcript", None) or r.get("text", None)
                if not d_raw or not content:
                    continue

                call_date = _call_date_key(d_raw)
                if not call_date:
                    continue

                if not in_window(call_date, args.start, args.end):
                    continue

                content_s = str(content)
                if call_date in by_date:
                    # prefer longer content if duplicates
                    _, existing_content, _, _ = by_date[call_date]
                    if len(content_s) <= len(existing_content):
                        continue

                by_date[call_date] = (r, content_s, year, q)

    all_dates_sorted = sorted(by_date.keys())
    if len(all_dates_sorted) > expected:
        keep_dates = set(all_dates_sorted[-expected:])
    else:
        keep_dates = set(all_dates_sorted)

    # Write kept transcripts
    saved = 0
    for call_date in sorted(keep_dates):
        r, content_s, year, q = by_date[call_date]

        td = out_base / call_date
        ensure_dir(td)

        write_json(r, td / "transcript.json")
        with open(td / "transcript.txt", "w", encoding="utf-8") as f:
            f.write(content_s)

        meta = {
            "ticker": ticker,
            "call_date": call_date,
            "fmp_year": year,
            "fmp_quarter": q,
            "downloaded_at_et": now_iso(),
            "source": "Financial Modeling Prep (FMP) earning_call_transcript endpoint",
            "expected": expected,
        }
        write_json(meta, td / "meta.json")
        saved += 1

    # Optionally prune extras on disk (safe for regenerated data; prevents lingering 21)
    prune = not args.no_prune_extra
    pruned = 0
    if prune and keep_dates:
        for child in _list_date_dirs(out_base):
            if child.name not in keep_dates:
                shutil.rmtree(child, ignore_errors=True)
                pruned += 1

    total_found = len(all_dates_sorted)
    print(
        f"[OK] Found {total_found} transcript dates; kept={len(keep_dates)} saved={saved} "
        f"(expected={expected}) prune={prune} pruned={pruned} into {out_base}"
    )


if __name__ == "__main__":
    main()
