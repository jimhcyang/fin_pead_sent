#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--fiscal-year-min", type=int, default=2020)
    ap.add_argument("--fiscal-year-max", type=int, default=2026)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    cfg = FMPConfig.from_env()
    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    out_base = data_dir / ticker / "transcripts"
    ensure_dir(out_base)

    saved = 0
    seen_dates = set()

    for year in range(args.fiscal_year_min, args.fiscal_year_max + 1):
        for q in [1, 2, 3, 4]:
            try:
                recs = fetch_one(cfg, ticker, year, q, args.sleep)
            except Exception:
                continue

            for r in recs:
                # common fields: date, content
                d = str(r.get("date", "")).strip()
                content = r.get("content", None) or r.get("transcript", None) or r.get("text", None)
                if not d or not content:
                    continue

                call_date = d.split(" ")[0]
                if not in_window(call_date, args.start, args.end):
                    continue
                if call_date in seen_dates:
                    continue
                seen_dates.add(call_date)

                # write under transcripts/YYYY-MM-DD/
                td = out_base / call_date
                ensure_dir(td)

                write_json(r, td / "transcript.json")
                with open(td / "transcript.txt", "w", encoding="utf-8") as f:
                    f.write(str(content))

                meta = {
                    "ticker": ticker,
                    "call_date": call_date,
                    "fmp_year": year,
                    "fmp_quarter": q,
                    "downloaded_at_et": now_iso(),
                    "source": "Financial Modeling Prep (FMP) earning_call_transcript endpoint",
                }
                write_json(meta, td / "meta.json")
                saved += 1

    print(f"[OK] Saved {saved} transcripts into {out_base}")


if __name__ == "__main__":
    main()
