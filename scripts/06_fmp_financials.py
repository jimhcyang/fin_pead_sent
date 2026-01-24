#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore

ET = ZoneInfo("America/New_York") if ZoneInfo else None


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class FMPConfig:
    api_key: str
    base_url: str = "https://financialmodelingprep.com"

    @staticmethod
    def from_env() -> "FMPConfig":
        key = os.getenv("FMP_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "Missing FMP_API_KEY in environment.\n"
                "Example:\n"
                "  export FMP_API_KEY='YOUR_KEY_HERE'\n"
            )
        return FMPConfig(api_key=key)


# -----------------------------
# Utils
# -----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _now_et_iso() -> str:
    if ET:
        return datetime.now(ET).isoformat()
    return datetime.utcnow().isoformat() + "Z"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# -----------------------------
# Fetch (with cache)
# -----------------------------
def _fetch_fmp_list(
    cfg: FMPConfig,
    endpoint: str,
    ticker: str,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    url = f"{cfg.base_url}/api/v3/{endpoint}/{ticker.upper()}"
    params = dict(params)
    params["apikey"] = cfg.api_key

    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {r.url}\n{r.text[:400]}")

    data = r.json()

    # FMP sometimes returns {"Error Message": "..."} payloads.
    if isinstance(data, dict) and any(k.lower().startswith("error") for k in data.keys()):
        raise RuntimeError(f"FMP error payload for {endpoint}/{ticker}: {data}")

    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected payload type for {endpoint}/{ticker}: {type(data)}")

    return data


def _load_or_fetch(
    cfg: FMPConfig,
    raw_path: Path,
    endpoint: str,
    ticker: str,
    params: Dict[str, Any],
    refresh: bool,
) -> List[Dict[str, Any]]:
    if raw_path.exists() and not refresh:
        return json.loads(raw_path.read_text(encoding="utf-8"))
    rows = _fetch_fmp_list(cfg, endpoint, ticker, params=params)
    _write_json(raw_path, rows)
    return rows


# -----------------------------
# Alignment source: transcripts meta
# -----------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _extract_fy_period_from_meta(meta: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Robustly extract (fiscalYear, period) from transcript meta.json.

    Your 04_fmp_transcripts.py writes:
      - {"fmp_year": 2024, "fmp_quarter": 1}

    We also support other common shapes:
      - {"year": 2024, "quarter": 4}
      - {"fiscalYear": "2024", "quarter": "Q4"}
      - {"calendarYear": 2024, "period": "Q4"}
    """
    # year candidates (YOUR pipeline uses fmp_year)
    fy = (
        meta.get("fiscalYear")
        or meta.get("calendarYear")
        or meta.get("year")
        or meta.get("fmp_year")
        or meta.get("fmpYear")
    )

    # quarter candidates (YOUR pipeline uses fmp_quarter)
    q = (
        meta.get("quarter")
        or meta.get("fiscalQuarter")
        or meta.get("q")
        or meta.get("fmp_quarter")
        or meta.get("fmpQuarter")
    )

    period = meta.get("period")

    # normalize fiscalYear -> string
    fy_s: Optional[str]
    if fy is None or (isinstance(fy, float) and pd.isna(fy)):
        fy_s = None
    else:
        try:
            fy_s = str(int(float(fy)))
        except Exception:
            fy_s = str(fy).strip() or None

    # normalize period
    if period is not None:
        p = str(period).strip().upper()
        if p in {"Q1", "Q2", "Q3", "Q4"}:
            return fy_s, p

    if q is None:
        return fy_s, None

    qs = str(q).strip().upper()
    if qs in {"Q1", "Q2", "Q3", "Q4"}:
        return fy_s, qs

    try:
        qi = int(float(qs))
        if qi in (1, 2, 3, 4):
            return fy_s, f"Q{qi}"
    except Exception:
        pass

    return fy_s, None


def _build_events_from_transcripts(transcripts_dir: Path, start_d: date, end_d: date) -> pd.DataFrame:
    """
    Events are defined by transcript folders: data/{TICKER}/transcripts/YYYY-MM-DD/
    We read each folder's meta.json to get (fiscalYear, period).
    """
    if not transcripts_dir.exists():
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for child in sorted(transcripts_dir.iterdir()):
        if not child.is_dir():
            continue
        if not _DATE_RE.match(child.name):
            continue

        earnings_date = _parse_yyyy_mm_dd(child.name)
        if not (start_d <= earnings_date <= end_d):
            continue

        meta_path = child / "meta.json"
        if not meta_path.exists():
            raise RuntimeError(f"Missing transcript meta.json: {meta_path}")

        meta = _read_json(meta_path)
        fy, period = _extract_fy_period_from_meta(meta)

        rows.append(
            {
                "earnings_date": earnings_date,
                "fiscalYear": fy,
                "period": period,
                "transcript_dir": str(child),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("earnings_date").reset_index(drop=True)
    return df


# -----------------------------
# Normalize / Align
# -----------------------------
def _normalize_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ---- KEY FIX #1: alias calendarYear -> fiscalYear for merge ----
    if "fiscalYear" not in df.columns:
        if "calendarYear" in df.columns:
            df["fiscalYear"] = df["calendarYear"]
        elif "year" in df.columns:
            df["fiscalYear"] = df["year"]
        else:
            df["fiscalYear"] = None

    # normalize fiscalYear as string
    df["fiscalYear"] = df["fiscalYear"].apply(
        lambda x: str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else (str(x).strip() if x is not None else None)
    ).astype("string")

    # period normalize
    if "period" in df.columns:
        df["period"] = df["period"].astype("string").str.upper()
    else:
        df["period"] = pd.Series([None] * len(df), dtype="string")

    # parse fiscal period end date (if present)
    if "date" in df.columns:
        df["_date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["_date_dt"] = pd.NaT

    # Deduplicate by (fiscalYear, period), keep latest by date
    df = (
        df.sort_values("_date_dt")
        .drop_duplicates(subset=["fiscalYear", "period"], keep="last")
        .reset_index(drop=True)
    )

    return df


def _align_table_to_events(table: pd.DataFrame, events: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if events.empty:
        aligned = pd.DataFrame()
        meta = {"ticker": ticker, "events": 0, "matched": 0, "missing_keys_in_events": 0, "missing_financial_rows": 0}
        return aligned, meta

    ev = events.copy()
    ev["fiscalYear"] = ev["fiscalYear"].astype("string")
    ev["period"] = ev["period"].astype("string").str.upper()

    aligned = ev.merge(table, on=["fiscalYear", "period"], how="left")

    # ensure symbol column
    if "symbol" in aligned.columns:
        aligned["symbol"] = aligned["symbol"].fillna(ticker)
    else:
        aligned.insert(0, "symbol", ticker)

    if "date" in aligned.columns:
        aligned = aligned.rename(columns={"date": "fiscal_period_end"})

    matched = int(aligned["fiscal_period_end"].notna().sum()) if "fiscal_period_end" in aligned.columns else 0

    meta = {
        "ticker": ticker,
        "events": int(events.shape[0]),
        "matched": matched,
        "missing_keys_in_events": int(events["fiscalYear"].isna().sum() + events["period"].isna().sum()),
        "missing_financial_rows": int(aligned["fiscal_period_end"].isna().sum()) if "fiscal_period_end" in aligned.columns else None,
    }
    return aligned, meta


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download FMP quarterly ratios + key metrics and ALIGN them to transcript earnings calls (1 row per earnings call)."
    )
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--save-as-is", action="store_true")
    args = ap.parse_args()

    ticker = args.ticker.upper().strip()
    start_d = _parse_yyyy_mm_dd(args.start)
    end_d = _parse_yyyy_mm_dd(args.end)

    cfg = FMPConfig.from_env()

    base_data = Path(args.data_dir) if args.data_dir else Path("data")
    tkr_dir = base_data / ticker
    fin_dir = tkr_dir / "financials"
    transcripts_dir = tkr_dir / "transcripts"
    _ensure_dir(fin_dir)

    events = _build_events_from_transcripts(transcripts_dir, start_d, end_d)
    if events.empty:
        raise RuntimeError(
            f"No transcript events found under {transcripts_dir} in [{start_d}..{end_d}].\n"
            f"Run transcripts first (04_fmp_transcripts.py), then re-run this script."
        )

    # ---- KEY FIX #2: fail fast if transcript events have missing keys ----
    bad = events[events["fiscalYear"].isna() | events["period"].isna()]
    if not bad.empty:
        sample = bad.head(5).to_string(index=False)
        raise RuntimeError(
            "Transcript events are missing fiscalYear/period (cannot align).\n"
            "This usually means transcript meta.json keys changed.\n"
            f"Sample bad rows:\n{sample}"
        )

    params = {"period": "quarter", "limit": int(args.limit)}

    ratios_raw = fin_dir / "ratios_quarter.raw.json"
    ratios_rows = _load_or_fetch(cfg, ratios_raw, "ratios", ticker, params=params, refresh=args.refresh)
    ratios_df = _normalize_table(ratios_rows)

    km_raw = fin_dir / "key_metrics_quarter.raw.json"
    km_rows = _load_or_fetch(cfg, km_raw, "key-metrics", ticker, params=params, refresh=args.refresh)
    km_df = _normalize_table(km_rows)

    if args.save_as_is:
        ratios_df.drop(columns=["_date_dt"], errors="ignore").to_csv(fin_dir / "ratios_quarter.as_is.csv", index=False)
        km_df.drop(columns=["_date_dt"], errors="ignore").to_csv(fin_dir / "key_metrics_quarter.as_is.csv", index=False)

    ratios_aligned, ratios_meta = _align_table_to_events(ratios_df, events, ticker)
    km_aligned, km_meta = _align_table_to_events(km_df, events, ticker)

    # Drop helper col
    for df in (ratios_aligned, km_aligned):
        df.drop(columns=["_date_dt"], errors="ignore", inplace=True)

    ratios_aligned.to_csv(fin_dir / "ratios_quarter.csv", index=False)
    km_aligned.to_csv(fin_dir / "key_metrics_quarter.csv", index=False)

    _write_json(
        fin_dir / "ratios_quarter.meta.json",
        {
            **ratios_meta,
            "endpoint": f"/api/v3/ratios/{ticker}",
            "params": params,
            "aligned_to": "transcripts",
            "downloaded_at_et": _now_et_iso(),
            "rows_saved": int(ratios_aligned.shape[0]),
        },
    )
    _write_json(
        fin_dir / "key_metrics_quarter.meta.json",
        {
            **km_meta,
            "endpoint": f"/api/v3/key-metrics/{ticker}",
            "params": params,
            "aligned_to": "transcripts",
            "downloaded_at_et": _now_et_iso(),
            "rows_saved": int(km_aligned.shape[0]),
        },
    )

    print(f"[OK] {ticker} financials aligned to transcripts:", flush=True)
    print(f"     events (transcripts): {events.shape[0]}", flush=True)
    print(f"     ratios_quarter.csv      rows={ratios_aligned.shape[0]}  matched={ratios_meta['matched']}", flush=True)
    print(f"     key_metrics_quarter.csv rows={km_aligned.shape[0]}  matched={km_meta['matched']}", flush=True)
    print(f"[OK] wrote -> {fin_dir}", flush=True)


if __name__ == "__main__":
    main()
