#!/usr/bin/env python3
# _eventlib.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


WINDOWS: Dict[str, Tuple[int, int]] = {
    "m5_p5": (-5, 5),
    "m1_p1": (-1, 1),
    "p0_p1": (0, 1),
    "p0_p5": (0, 5),
    "p0_p10": (0, 10),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_data_dir() -> Path:
    return repo_root() / "data"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _date_str(d: date) -> str:
    return pd.to_datetime(d).strftime("%Y-%m-%d")


def load_prices(data_dir: Path, ticker: str) -> pd.DataFrame:
    """
    Loads data/{T}/prices/yf_ohlcv_daily.csv.
    Requires: date, adj_close
    Tolerates extra columns (open/high/low/close/volume).
    Returns: df indexed by Timestamp(date), sorted.
    """
    p = data_dir / ticker / "prices" / "yf_ohlcv_daily.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing prices: {p}")

    df = pd.read_csv(p)
    if "date" not in df.columns:
        raise ValueError(f"{p} missing column 'date'")
    if "adj_close" not in df.columns:
        # tolerate alt schema
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "adj_close"})
        else:
            raise ValueError(f"{p} missing column 'adj_close'")

    df["date"] = _to_dt(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # numeric coercions
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.set_index("date").sort_index()
    return df


def load_market(data_dir: Path, rel: str = "_tmp_market/spx/prices/yf_ohlcv_daily.csv") -> pd.DataFrame:
    p = data_dir / rel
    if not p.exists():
        raise FileNotFoundError(f"Missing market file: {p}")
    df = pd.read_csv(p)
    if "date" not in df.columns or "adj_close" not in df.columns:
        raise ValueError(f"Market file must have columns date, adj_close: {p}")
    df["date"] = _to_dt(df["date"])
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["date", "adj_close"]).sort_values("date").set_index("date")
    return df


def load_calendar(data_dir: Path, ticker: str) -> pd.DataFrame:
    p = data_dir / ticker / "calendar" / "earnings_calendar.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing calendar: {p}")
    df = pd.read_csv(p)
    if "earnings_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "earnings_date"})
    if "earnings_date" not in df.columns:
        raise ValueError(f"{p} missing earnings_date")
    df["earnings_date"] = _to_date(df["earnings_date"])
    df = df.dropna(subset=["earnings_date"]).sort_values("earnings_date").reset_index(drop=True)

    # infer timing col
    timing_col = None
    for c in ["announce_timing", "time", "timing", "when"]:
        if c in df.columns:
            timing_col = c
            break
    if timing_col is None:
        df["announce_timing"] = "amc"
        timing_col = "announce_timing"

    df["announce_timing"] = df[timing_col].astype(str).str.strip().str.lower()
    df.loc[~df["announce_timing"].isin(["bmo", "amc"]), "announce_timing"] = "amc"
    return df[["earnings_date", "announce_timing"]].copy()


def _asof_index(trading: List[pd.Timestamp], d: date) -> int:
    """
    last trading date <= d (calendar date).
    raises if d before first trading day.
    """
    td = pd.to_datetime(d)
    if td < trading[0]:
        raise ValueError(f"date {d} before first trading day {trading[0].date()}")
    lo, hi = 0, len(trading) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if trading[mid] <= td:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi


def _idx(trading: List[pd.Timestamp], i: int) -> Optional[pd.Timestamp]:
    if i < 0 or i >= len(trading):
        return None
    return trading[i]


def build_event_windows_df(
    data_dir: Path,
    ticker: str,
    pre_bdays: int = 5,
    post_bdays: int = 10,
) -> pd.DataFrame:
    """
    Day-0 convention for RETURNS:
      day0 = react_close_date:
        - BMO: same-day close (asof earnings_date)
        - AMC: next trading-day close

    Also stores:
      pre_close_date = close immediately BEFORE announcement.
    """
    tkr = ticker.upper()
    px = load_prices(data_dir, tkr)
    cal = load_calendar(data_dir, tkr)
    trading = list(px.index)  # sorted Timestamps

    rows = []
    for _, r in cal.iterrows():
        ed: date = r["earnings_date"]
        timing = str(r["announce_timing"]).lower()

        i = _asof_index(trading, ed)

        if timing == "bmo":
            pre_i = i - 1
            day0_i = i
        else:
            # amc
            pre_i = i
            day0_i = i + 1

        pre_dt = _idx(trading, pre_i)
        day0_dt = _idx(trading, day0_i)

        if pre_dt is None or day0_dt is None:
            continue

        # offsets relative to day0
        def off(k: int) -> Optional[pd.Timestamp]:
            return _idx(trading, day0_i + k)

        dt_m5 = off(-5)
        dt_m1 = off(-1)
        dt_p1 = off(1)
        dt_p5 = off(5)
        dt_p10 = off(10)
        win_start = off(-pre_bdays)
        win_end = off(post_bdays)

        need = [dt_m5, dt_m1, day0_dt, dt_p1, dt_p5, dt_p10, win_start, win_end]
        if any(x is None for x in need):
            continue

        rows.append(
            {
                "ticker": tkr,
                "earnings_date": _date_str(ed),
                "announce_timing": timing,
                "pre_close_date": pre_dt.strftime("%Y-%m-%d"),
                "day0_date": day0_dt.strftime("%Y-%m-%d"),

                "dt_m5": dt_m5.strftime("%Y-%m-%d"),
                "dt_m1": dt_m1.strftime("%Y-%m-%d"),
                "dt_p1": dt_p1.strftime("%Y-%m-%d"),
                "dt_p5": dt_p5.strftime("%Y-%m-%d"),
                "dt_p10": dt_p10.strftime("%Y-%m-%d"),

                "path_start": win_start.strftime("%Y-%m-%d"),
                "path_end": win_end.strftime("%Y-%m-%d"),

                "pre_bdays": int(pre_bdays),
                "post_bdays": int(post_bdays),
            }
        )

    return pd.DataFrame(rows).sort_values("earnings_date").reset_index(drop=True)


def extract_event_price_path_long(
    data_dir: Path,
    ticker: str,
    windows_csv_name: str = "event_windows",
    out_pre_bdays: int = 5,
    out_post_bdays: int = 10,
) -> pd.DataFrame:
    tkr = ticker.upper()
    px = load_prices(data_dir, tkr)

    wpath = data_dir / tkr / "events" / f"{windows_csv_name}.csv"
    if not wpath.exists():
        raise FileNotFoundError(f"Missing windows file: {wpath} (run script 11)")

    w = pd.read_csv(wpath)
    if "day0_date" not in w.columns:
        raise ValueError(f"{wpath} missing day0_date")

    rows = []
    for _, ev in w.iterrows():
        ed = ev["earnings_date"]
        timing = ev.get("announce_timing", "amc")
        day0 = pd.to_datetime(ev["day0_date"], errors="coerce")
        if pd.isna(day0):
            continue

        for k in range(-out_pre_bdays, out_post_bdays + 1):
            d = day0 + pd.tseries.offsets.BDay(k)  # NOTE: will not match holidays; we'll map via px index below
            # Use trading-day indexing (px index), not weekday-only:
            # Find position of day0 in px.index, then offset by k.
        # trading index approach:
        if day0 not in px.index:
            # if market close date mismatch, skip
            continue
        i0 = int(px.index.get_indexer([day0])[0])
        for k in range(-out_pre_bdays, out_post_bdays + 1):
            j = i0 + k
            if j < 0 or j >= len(px.index):
                continue
            dt = px.index[j]
            rec = {
                "ticker": tkr,
                "earnings_date": ed,
                "announce_timing": timing,
                "day0_date": ev["day0_date"],
                "date": dt.strftime("%Y-%m-%d"),
                "offset": int(k),
            }
            # attach price columns if present
            for c in ["open", "high", "low", "close", "adj_close", "volume"]:
                if c in px.columns:
                    rec[c] = float(px.loc[dt, c]) if pd.notna(px.loc[dt, c]) else np.nan
            rows.append(rec)

    out = pd.DataFrame(rows)
    return out.sort_values(["ticker", "earnings_date", "offset"]).reset_index(drop=True)


def compute_event_daily_returns(
    price_path_long: pd.DataFrame,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """
    Returns a LONG df:
      ticker, earnings_date, date, offset, ret (decimal), ret_pct
    ret at offset k is (P_k/P_{k-1} - 1) for k > min_offset.
    """
    req = {"ticker", "earnings_date", "offset", price_col}
    missing = req - set(price_path_long.columns)
    if missing:
        raise ValueError(f"price_path_long missing required cols: {sorted(missing)}")

    df = price_path_long.copy()
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    rows = []
    for (tkr, ed), g in df.groupby(["ticker", "earnings_date"], sort=True):
        g = g.sort_values("offset").reset_index(drop=True)
        # compute close-to-close returns
        p = g[price_col].to_numpy(dtype=float)
        offs = g["offset"].to_numpy(dtype=int)

        ret = np.full_like(p, np.nan, dtype=float)
        for i in range(1, len(p)):
            if np.isfinite(p[i]) and np.isfinite(p[i - 1]) and p[i - 1] != 0:
                ret[i] = p[i] / p[i - 1] - 1.0

        for i in range(len(g)):
            rows.append(
                {
                    "ticker": tkr,
                    "earnings_date": ed,
                    "date": g.loc[i, "date"] if "date" in g.columns else None,
                    "offset": int(offs[i]),
                    "ret": float(ret[i]) if np.isfinite(ret[i]) else np.nan,
                    "ret_pct": float(100.0 * ret[i]) if np.isfinite(ret[i]) else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(["ticker", "earnings_date", "offset"]).reset_index(drop=True)


def compute_event_window_pct_changes(
    price_path_long: pd.DataFrame,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """
    For each event, compute pct change over WINDOWS using endpoint prices:
      pct = 100*(P_end/P_start - 1)
    """
    req = {"ticker", "earnings_date", "offset", price_col}
    missing = req - set(price_path_long.columns)
    if missing:
        raise ValueError(f"price_path_long missing required cols: {sorted(missing)}")

    df = price_path_long.copy()
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    rows = []
    for (tkr, ed), g in df.groupby(["ticker", "earnings_date"], sort=True):
        mp = dict(zip(g["offset"].astype(int).tolist(), g[price_col].tolist()))

        rec = {"ticker": tkr, "earnings_date": ed}
        # store key endpoint prices
        for k in [-5, -1, 0, 1, 5, 10]:
            rec[f"px_{k:+d}"] = float(mp.get(k, np.nan))

        for name, (a, b) in WINDOWS.items():
            pa = float(mp.get(a, np.nan))
            pb = float(mp.get(b, np.nan))
            if np.isfinite(pa) and np.isfinite(pb) and pa != 0:
                rec[f"pct_{name}"] = 100.0 * (pb / pa - 1.0)
                rec[f"logpct_{name}"] = 100.0 * float(np.log(pb / pa))
            else:
                rec[f"pct_{name}"] = np.nan
                rec[f"logpct_{name}"] = np.nan

        rows.append(rec)

    return pd.DataFrame(rows).sort_values(["ticker", "earnings_date"]).reset_index(drop=True)


def fit_alpha_beta(ri: np.ndarray, rm: np.ndarray) -> Tuple[float, float, float, float, int]:
    """
    Market model: ri = alpha + beta*rm + eps
    Inputs ri/rm are daily returns in decimal.
    Returns: alpha, beta, r2, resid_std, n
    """
    m = np.isfinite(ri) & np.isfinite(rm)
    ri = ri[m]
    rm = rm[m]
    n = int(len(ri))
    if n < 30:
        return np.nan, np.nan, np.nan, np.nan, n

    vrm = float(np.var(rm, ddof=1)) if n > 1 else 0.0
    if not np.isfinite(vrm) or vrm <= 1e-18:
        return np.nan, np.nan, np.nan, np.nan, n

    cov = float(np.cov(ri, rm, ddof=1)[0, 1])
    beta = cov / vrm
    alpha = float(np.mean(ri) - beta * np.mean(rm))

    resid = ri - (alpha + beta * rm)
    resid_std = float(np.std(resid, ddof=1)) if n > 1 else np.nan

    vri = float(np.var(ri, ddof=1)) if n > 1 else np.nan
    r2 = float(1.0 - (np.var(resid, ddof=1) / vri)) if (np.isfinite(vri) and vri > 1e-18) else np.nan

    return alpha, beta, r2, resid_std, n


def window_return_offsets(a: int, b: int) -> List[int]:
    """
    Returns offsets whose returns compose the move from a -> b.
    If prices are defined at offsets, then returns are at offsets (a+1..b).
    """
    if b <= a:
        return []
    return list(range(a + 1, b + 1))


def compute_window_car(abn_by_offset: Dict[int, float], a: int, b: int) -> float:
    """Sum abnormal returns over window [a,b] using offset-labeled daily abnormal returns.

    By convention, if returns are labeled by the *day they occur on* (offset t),
    then CAR[a,b] is the sum over offsets in [a,b].

    This function is intentionally tolerant of missing offsets near the edges,
    but MUST work for short windows like [0,1] and [-1,1].
    """
    offs = window_return_offsets(a, b)
    vals = np.array([abn_by_offset.get(k, np.nan) for k in offs], dtype=float)
    L = int(len(vals))
    if L == 0:
        return np.nan

    finite_n = int(np.isfinite(vals).sum())

    # For short windows (L<=2), require full coverage; otherwise allow some missingness.
    # (The previous >=3 rule made CAR for 1–2 day windows always NaN.)
    min_required = L if L <= 2 else max(3, int(np.ceil(0.5 * L)))
    if finite_n < min_required:
        return np.nan

    return float(np.nansum(vals))


def save_with_meta(df: pd.DataFrame, out_csv: Path, meta: Dict) -> None:
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    meta = dict(meta)
    meta["rows"] = int(df.shape[0])
    meta["cols"] = int(df.shape[1])
    meta["created_at_local"] = datetime.now().isoformat()
    (out_csv.parent / (out_csv.stem + ".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
