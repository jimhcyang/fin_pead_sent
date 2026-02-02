#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
29_build_event_study_panel.py

Build a proper per-event panel using transcript dates as event dates.

Definitions (your spec)
- Estimation window: (-260, -11) relative to t(-1)
- Reaction window: (-1, +1)
- Drift window: (+2, -2N) where -2N = close 2 business days before next event

Outputs:
data/{TICKER}/events/event_study_panel.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class EventRow:
    ticker: str
    event_idx: int
    event_date: pd.Timestamp
    next_event_date: Optional[pd.Timestamp]

    pre_date: Optional[pd.Timestamp]          # t(-1)
    post1_date: Optional[pd.Timestamp]        # t(+1)
    drift_start_date: Optional[pd.Timestamp]  # t(+2)
    drift_end_date: Optional[pd.Timestamp]    # t(-2N)

    est_start_date: Optional[pd.Timestamp]    # t(-260)
    est_end_date: Optional[pd.Timestamp]      # t(-11)


def _resolve_market_path(market_path: Path, data_dir: Path) -> Path:
    """
    Robust resolution for market prices file:

    - If market_path exists as given -> use it.
    - Else try data_dir / market_path (common when passing '_tmp_market/...' but file is in 'data/_tmp_market/...').
    """
    mp = Path(market_path)

    if mp.exists():
        return mp

    cand = Path(data_dir) / mp
    if cand.exists():
        return cand

    # return original so downstream raises clear FileNotFoundError with the exact path it looked for
    return mp


def _load_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prices file: {path}")

    df = pd.read_csv(path)

    date_col = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
    if date_col is None:
        raise ValueError(f"No date column found in {path}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).rename(columns={date_col: "date"})

    # tolerate YF schemas
    if "Adj Close" in df.columns and "adj_close" not in df.columns:
        df = df.rename(columns={"Adj Close": "adj_close"})

    if "adj_close" not in df.columns:
        raise ValueError(f"Expected adj_close in {path.name}")

    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["adj_close"]).set_index("date").sort_index()
    return df


def _load_event_dates_from_transcripts(ticker_dir: Path) -> List[pd.Timestamp]:
    tdir = ticker_dir / "transcripts"
    if not tdir.exists():
        return []

    dates: List[pd.Timestamp] = []
    for p in tdir.iterdir():
        if p.is_dir():
            try:
                dates.append(pd.to_datetime(p.name))
            except Exception:
                pass
    return sorted(dates)


def _load_event_dates_from_calendar(ticker_dir: Path) -> List[pd.Timestamp]:
    cal = ticker_dir / "calendar" / "earnings_calendar.csv"
    if not cal.exists():
        return []

    df = pd.read_csv(cal)
    col = "earnings_date" if "earnings_date" in df.columns else ("date" if "date" in df.columns else None)
    if col is None:
        return []

    s = pd.to_datetime(df[col], errors="coerce").dropna()
    return sorted(s.unique().tolist())


def _prev_trading_day(trading_days: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = trading_days.searchsorted(dt, side="left") - 1
    if pos < 0:
        return None
    return trading_days[pos]


def _next_trading_day(trading_days: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = trading_days.searchsorted(dt, side="right")
    if pos >= len(trading_days):
        return None
    return trading_days[pos]


def _fit_market_model(stock_rets: pd.Series, mkt_rets: pd.Series) -> Tuple[float, float, float, float]:
    """Return alpha, beta, r2, resid_std (all in daily-return units)."""
    df = pd.concat([stock_rets, mkt_rets], axis=1).dropna()
    if len(df) < 30:
        return (np.nan, np.nan, np.nan, np.nan)

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1].values.reshape(-1, 1)

    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)

    alpha = float(lr.intercept_)
    beta = float(lr.coef_[0])
    r2 = float(lr.score(X, y))

    resid = y - lr.predict(X)
    resid_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else np.nan

    return alpha, beta, r2, resid_std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--market-prices", type=Path, default=Path("_tmp_market/spx/prices/yf_ohlcv_daily.csv"))
    ap.add_argument("--est-len", type=int, default=250)   # (-260..-11) inclusive => 250 trading days
    ap.add_argument("--est-gap", type=int, default=10)    # t(-11) is 10 trading days before t(-1)
    ap.add_argument(
        "--keep-last-event",
        action="store_true",
        help="Keep last event row even though drift end is undefined (target will be NaN).",
    )
    args = ap.parse_args()

    tkr = args.ticker.upper()
    tdir = args.data_dir / tkr

    px = _load_prices(tdir / "prices" / "yf_ohlcv_daily.csv")

    market_path = _resolve_market_path(args.market_prices, args.data_dir)
    mkt = _load_prices(market_path)

    trading_days = px.index

    # daily returns series (return dated at day t)
    _ = px["adj_close"].pct_change()
    _ = mkt["adj_close"].pct_change()

    # events: transcript dates first, calendar fallback
    event_dates = _load_event_dates_from_transcripts(tdir)
    if not event_dates:
        event_dates = _load_event_dates_from_calendar(tdir)
    if not event_dates:
        raise FileNotFoundError(f"No transcript dates or earnings_calendar.csv for {tkr}")

    rows: List[dict] = []
    for i, ev in enumerate(event_dates):
        next_ev = event_dates[i + 1] if i + 1 < len(event_dates) else None

        pre = _prev_trading_day(trading_days, ev)
        post1 = _next_trading_day(trading_days, ev)

        drift_start = None
        if post1 is not None:
            j = trading_days.get_indexer([post1])[0]
            if j >= 0 and j + 1 < len(trading_days):
                drift_start = trading_days[j + 1]  # +2

        drift_end = None
        if next_ev is not None:
            next_pre = _prev_trading_day(trading_days, next_ev)
            if next_pre is not None:
                k = trading_days.get_indexer([next_pre])[0]
                if k - 1 >= 0:
                    drift_end = trading_days[k - 1]  # -2N

        # estimation window dates
        est_start = est_end = None
        if pre is not None:
            pre_pos = trading_days.get_indexer([pre])[0]
            est_end_pos = pre_pos - args.est_gap
            est_start_pos = est_end_pos - (args.est_len - 1)
            if est_start_pos >= 0:
                est_end = trading_days[est_end_pos]
                est_start = trading_days[est_start_pos]

        # optionally drop last event row (no drift end)
        if next_ev is None and not args.keep_last_event:
            continue

        rows.append(
            {
                "ticker": tkr,
                "event_idx": i + 1,
                "event_date": ev,
                "next_event_date": next_ev,
                "pre_date": pre,
                "post1_date": post1,
                "drift_start_date": drift_start,
                "drift_end_date": drift_end,
                "est_start_date": est_start,
                "est_end_date": est_end,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No valid events built for {tkr}")

    out_rows = []
    for r in df.to_dict(orient="records"):
        pre = r["pre_date"]
        post1 = r["post1_date"]
        d0 = r["drift_start_date"]
        d1 = r["drift_end_date"]
        e0 = r["est_start_date"]
        e1 = r["est_end_date"]

        rec = dict(r)

        # prices
        def p(dt: Optional[pd.Timestamp]) -> float:
            if dt is None or pd.isna(dt):
                return np.nan
            return float(px.loc[dt, "adj_close"]) if dt in px.index else np.nan

        rec["pre_price"] = p(pre)
        rec["post1_price"] = p(post1)
        rec["drift_start_price"] = p(d0)
        rec["drift_end_price"] = p(d1)

        # reaction return (-1, +1)
        if np.isfinite(rec["pre_price"]) and np.isfinite(rec["post1_price"]):
            rec["reaction_ret_pct"] = 100.0 * (rec["post1_price"] / rec["pre_price"] - 1.0)
        else:
            rec["reaction_ret_pct"] = np.nan

        # estimation stats + market model params
        if e0 is not None and e1 is not None and (e0 in px.index) and (e1 in px.index) and (e0 < e1):
            est_px = px.loc[e0:e1, "adj_close"]
            est_stock_rets = est_px.pct_change().dropna()

            est_mkt = mkt.loc[e0:e1, "adj_close"] if (e0 in mkt.index and e1 in mkt.index) else None
            est_mkt_rets = est_mkt.pct_change().dropna() if est_mkt is not None else pd.Series(dtype=float)

            rec["est_n_days"] = int(len(est_stock_rets))
            rec["est_mean"] = float(est_stock_rets.mean()) if len(est_stock_rets) else np.nan
            rec["est_vol"] = float(est_stock_rets.std(ddof=1)) if len(est_stock_rets) > 1 else np.nan
            rec["est_skew"] = float(est_stock_rets.skew()) if len(est_stock_rets) > 2 else np.nan
            rec["est_kurt"] = float(est_stock_rets.kurt()) if len(est_stock_rets) > 3 else np.nan

            alpha, beta, r2, resid_std = _fit_market_model(est_stock_rets, est_mkt_rets)
            rec["alpha_mm"] = alpha
            rec["beta_mm"] = beta
            rec["r2_mm"] = r2
            rec["resid_vol_mm"] = resid_std
        else:
            rec["est_n_days"] = 0
            rec["est_mean"] = rec["est_vol"] = rec["est_skew"] = rec["est_kurt"] = np.nan
            rec["alpha_mm"] = rec["beta_mm"] = rec["r2_mm"] = rec["resid_vol_mm"] = np.nan

        # drift cumret + CAR
        if d0 is not None and d1 is not None and (d0 in px.index) and (d1 in px.index) and (d0 < d1):
            drift_px = px.loc[d0:d1, "adj_close"]
            drift_stock_rets = drift_px.pct_change().dropna()
            rec["drift_n_days"] = int(len(drift_stock_rets))

            if (d0 in mkt.index) and (d1 in mkt.index):
                drift_mkt_px = mkt.loc[d0:d1, "adj_close"]
                drift_mkt_rets = drift_mkt_px.pct_change().dropna()
            else:
                drift_mkt_px = None
                drift_mkt_rets = pd.Series(dtype=float)

            drift_stock_cum = float(drift_px.iloc[-1] / drift_px.iloc[0] - 1.0)
            rec["drift_ret_pct"] = 100.0 * drift_stock_cum

            if drift_mkt_px is not None and len(drift_mkt_rets) > 0:
                drift_mkt_cum = float(drift_mkt_px.iloc[-1] / drift_mkt_px.iloc[0] - 1.0)
                rec["drift_mkt_ret_pct"] = 100.0 * drift_mkt_cum
                rec["drift_abret_simple_pct"] = 100.0 * (drift_stock_cum - drift_mkt_cum)
            else:
                rec["drift_mkt_ret_pct"] = np.nan
                rec["drift_abret_simple_pct"] = np.nan

            # market-model CAR
            if np.isfinite(rec["alpha_mm"]) and np.isfinite(rec["beta_mm"]) and len(drift_mkt_rets) > 0:
                dd = pd.concat([drift_stock_rets, drift_mkt_rets], axis=1).dropna()
                dd.columns = ["ri", "rm"]
                ar = dd["ri"] - (rec["alpha_mm"] + rec["beta_mm"] * dd["rm"])
                rec["drift_car_mm_pct"] = 100.0 * float(ar.sum())
            else:
                rec["drift_car_mm_pct"] = np.nan
        else:
            rec["drift_n_days"] = 0
            rec["drift_ret_pct"] = np.nan
            rec["drift_mkt_ret_pct"] = np.nan
            rec["drift_abret_simple_pct"] = np.nan
            rec["drift_car_mm_pct"] = np.nan

        out_rows.append(rec)

    out = pd.DataFrame(out_rows)

    out_dir = tdir / "events"
    ensure_dir(out_dir)
    out_path = out_dir / "event_study_panel.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] {tkr}: wrote {out_path} | rows={len(out)}")
    print(f"     drift target non-missing: {out['drift_car_mm_pct'].notna().sum()}")


if __name__ == "__main__":
    main()
