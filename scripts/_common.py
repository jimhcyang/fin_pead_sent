#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ---------------------------------------------------------------------
# Default tickers (your core 20)
# ---------------------------------------------------------------------
DEFAULT_20 = [
    "NVDA", "GOOGL", "AAPL", "MSFT", "AMZN",
    "META", "AVGO", "TSLA", "LLY", "WMT",
    "JPM", "V", "XOM", "JNJ", "ORCL",
    "MA", "MU", "COST", "AMD", "ABBV",
]


def repo_root() -> Path:
    # scripts/ is at repo_root/scripts
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    return repo_root() / "data"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(tz=ET).isoformat()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    ensure_dir(path.parent)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_dt_any(s: str, assume_tz: ZoneInfo = UTC) -> datetime:
    """
    Parse datetimes like:
      - '2024-11-20 15:45:00' (naive)
      - '2024-11-20' (date only)
      - ISO with offset
      - '...Z'
    If naive, assume assume_tz (default UTC).
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("parse_dt_any: empty datetime string")

    dt: Optional[datetime] = None

    # ISO-ish
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
    except Exception:
        dt = None

    # Common fallbacks
    if dt is None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except Exception:
                continue

    if dt is None:
        raise ValueError(f"Could not parse datetime: {s}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=assume_tz)

    return dt


def to_et(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ET)


@dataclass
class FMPConfig:
    api_key: str
    # Used by transcripts endpoint in 04: {base_url}/v3/earning_call_transcript/{ticker}
    base_url: str = "https://financialmodelingprep.com/api"

    @staticmethod
    def from_env() -> "FMPConfig":
        key = os.environ.get("FMP_API_KEY", "").strip()
        if not key:
            raise RuntimeError("Missing FMP_API_KEY in environment.")
        return FMPConfig(api_key=key)


def http_get_json(url: str, params: Dict[str, Any], sleep_s: float = 0.2) -> Any:
    import requests  # local import

    r = requests.get(url, params=params, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} for {r.url}\n{r.text[:400]}")
    time.sleep(sleep_s)  # gentle throttle
    return r.json()
