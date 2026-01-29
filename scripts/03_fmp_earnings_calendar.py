#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, date, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo  # py3.9+
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


ET = ZoneInfo("America/New_York") if ZoneInfo else None


@dataclass(frozen=True)
class FMPConfig:
    api_key: str
    base_url: str = "https://financialmodelingprep.com"

    @staticmethod
    def from_env() -> "FMPConfig":
        key = os.getenv("FMP_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "Missing FMP_API_KEY in environment. Example:\n"
                "  export FMP_API_KEY='YOUR_KEY_HERE'\n"
            )
        return FMPConfig(api_key=key)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _event_dt_placeholder(d: date, timing: str) -> Optional[str]:
    """
    Coarse ET timestamp for ordering only.
    - bmo: 08:00 ET
    - amc: 16:30 ET
    - otherwise: 12:00 ET
    """
    if ET is None:
        return None

    t = timing.lower().strip()
    if t == "bmo":
        tm = time(8, 0)
    elif t == "amc":
        tm = time(16, 30)
    else:
        tm = time(12, 0)

    dt_et = datetime(d.year, d.month, d.day, tm.hour, tm.minute, tzinfo=ET)
    return dt_et.isoformat()


def fetch_company_earning_calendar(cfg: FMPConfig, ticker: str) -> List[Dict[str, Any]]:
    """
    Company-specific earnings calendar (legacy endpoint):
      /api/v3/historical/earning_calendar/{SYMBOL}
    """
    url = f"{cfg.base_url}/api/v3/historical/earning_calendar/{ticker.upper()}"
    params = {"apikey": cfg.api_key}

    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"FMP HTTP {r.status_code}: {r.text[:500]}")

    data = r.json()

    # FMP sometimes returns {"Error Message": "..."} style payloads.
    if isinstance(data, dict) and any(k.lower().startswith("error") for k in data.keys()):
        raise RuntimeError(f"FMP error payload: {data}")

    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected payload type: {type(data)}")

    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--expected", type=int, default=20, help="Keep only the last N events (default 20).")
    ap.add_argument("--data-dir", default=None, help="Default: repo_root/data")

    args = ap.parse_args()

    ticker = args.ticker.upper()
    start_d = _parse_yyyy_mm_dd(args.start)
    end_d = _parse_yyyy_mm_dd(args.end)
    expected = int(args.expected)

    cfg = FMPConfig.from_env()

    base_dir = Path(args.data_dir) if args.data_dir else Path("data")
    out_dir = base_dir / ticker / "calendar"
    _ensure_dir(out_dir)

    raw = fetch_company_earning_calendar(cfg, ticker)

    # Save raw (as returned)
    raw_path = out_dir / "earnings_calendar.raw.json"
    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    # Normalize to dataframe
    df = pd.DataFrame(raw)

    # Defensive: some payloads might omit symbol; ensure ticker column is correct
    if "symbol" in df.columns:
        df["ticker"] = df["symbol"].astype(str)
    else:
        df["ticker"] = ticker

    # Date field
    if "date" not in df.columns:
        raise RuntimeError(f"Expected 'date' in payload, got columns: {list(df.columns)}")

    df["earnings_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Filter to requested window
    df = df[df["earnings_date"].between(start_d, end_d)].copy()

    # Timing
    timing_col = "time" if "time" in df.columns else None
    if timing_col:
        df["announce_timing"] = (
            df[timing_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": "--", "none": "--", "": "--"})
        )
    else:
        df["announce_timing"] = "--"

    # EPS / Revenue fields (keep as numeric)
    def _num(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([pd.NA] * len(df))

    df["eps_est"] = _num("epsEstimated")
    df["eps_actual"] = _num("eps")
    df["revenue_est"] = _num("revenueEstimated")
    df["revenue_actual"] = _num("revenue")

    # Placeholder datetime in ET (ordering only)
    df["event_datetime_et_placeholder"] = [
        _event_dt_placeholder(d, t)
        for d, t in zip(df["earnings_date"], df["announce_timing"])
    ]

    # Final columns
    out_cols = [
        "ticker",
        "earnings_date",
        "announce_timing",
        "eps_est",
        "eps_actual",
        "revenue_est",
        "revenue_actual",
        "event_datetime_et_placeholder",
    ]

    # Sort oldest -> newest, then keep last N (prevents MU=21)
    df_out = df[out_cols].sort_values(
        ["earnings_date", "event_datetime_et_placeholder"],
        ascending=True,
        kind="mergesort",
    )

    rows_raw_in_window = int(df_out.shape[0])
    trimmed = False
    if rows_raw_in_window > expected:
        df_out = df_out.tail(expected).copy()
        trimmed = True

    df_out = df_out.reset_index(drop=True)

    csv_path = out_dir / "earnings_calendar.csv"
    df_out.to_csv(csv_path, index=False)

    meta = {
        "ticker": ticker,
        "source": "FMP legacy endpoint: /api/v3/historical/earning_calendar/{symbol}",
        "start": str(start_d),
        "end": str(end_d),
        "downloaded_at_et": datetime.now(ET).isoformat() if ET else datetime.utcnow().isoformat() + "Z",
        "rows_raw_in_window": rows_raw_in_window,
        "rows_saved": int(df_out.shape[0]),
        "expected": expected,
        "trimmed_to_expected": trimmed,
        "trim_policy": "sorted_by_earnings_date_take_last_expected" if trimmed else None,
        "notes": "announce_timing is typically 'amc' or 'bmo'; ET placeholder is for ordering only.",
    }
    meta_path = out_dir / "earnings_calendar.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Wrote: {csv_path} ({df_out.shape[0]} rows)")
    print(f"[OK] Wrote: {raw_path}")
    print(f"[OK] Wrote: {meta_path}")


if __name__ == "__main__":
    main()
