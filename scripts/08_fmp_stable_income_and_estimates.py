# scripts/08_fmp_stable_income_and_estimates.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from _common import ET, FMPConfig, default_data_dir, ensure_dir, http_get_json, write_json

STABLE_BASE = "https://financialmodelingprep.com/stable"


def _now_et_iso() -> str:
    return datetime.now(ET).isoformat()


def _to_df(payload: Any) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return pd.DataFrame(payload["data"])
        return pd.DataFrame([payload])
    return pd.DataFrame()


def fetch_stable_endpoint(endpoint: str, params: Dict[str, Any], sleep_s: float) -> Any:
    url = f"{STABLE_BASE}/{endpoint}"
    return http_get_json(url, params=params, sleep_s=sleep_s)


def _prep_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out["_date_dt"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["_date_dt"]).sort_values("_date_dt", kind="mergesort")
    return out


def _filter_cutoff(df: pd.DataFrame, cutoff_end: Optional[str]) -> pd.DataFrame:
    if df.empty or cutoff_end is None:
        return df
    c = pd.to_datetime(cutoff_end, errors="coerce")
    if pd.isna(c):
        return df
    if "_date_dt" not in df.columns:
        return df
    return df[df["_date_dt"] <= c].copy()


def _tail_anchor_dates(df_stmt: pd.DataFrame, tail_n: int) -> List[pd.Timestamp]:
    if df_stmt.empty or "_date_dt" not in df_stmt.columns:
        return []
    dates = df_stmt["_date_dt"].drop_duplicates().sort_values()
    if tail_n is not None and tail_n > 0:
        dates = dates.tail(tail_n)
    return list(dates)


def _align_exact(df: pd.DataFrame, anchor_dates: List[pd.Timestamp]) -> pd.DataFrame:
    if not anchor_dates:
        return pd.DataFrame()

    anchors = pd.DataFrame({"_anchor_dt": anchor_dates})
    if df.empty or "_date_dt" not in df.columns:
        out = anchors.copy()
        out["date"] = out["_anchor_dt"].dt.strftime("%Y-%m-%d")
        return out.drop(columns=["_anchor_dt"])

    tmp = df.copy()
    tmp = tmp.drop_duplicates(subset=["_date_dt"], keep="last").sort_values("_date_dt", kind="mergesort")

    merged = anchors.merge(tmp, how="left", left_on="_anchor_dt", right_on="_date_dt")
    merged["date"] = merged["_anchor_dt"].dt.strftime("%Y-%m-%d")
    merged = merged.drop(columns=["_anchor_dt", "_date_dt"], errors="ignore")
    return merged


def _align_nearest(df: pd.DataFrame, anchor_dates: List[pd.Timestamp], window_days: int) -> pd.DataFrame:
    if not anchor_dates:
        return pd.DataFrame()

    anchors = pd.DataFrame({"_anchor_dt": anchor_dates}).sort_values("_anchor_dt", kind="mergesort")
    anchors["date"] = anchors["_anchor_dt"].dt.strftime("%Y-%m-%d")

    if df.empty or "_date_dt" not in df.columns:
        return anchors.drop(columns=["_anchor_dt"])

    tmp = df.copy().sort_values("_date_dt", kind="mergesort")
    merged = pd.merge_asof(
        anchors,
        tmp,
        left_on="_anchor_dt",
        right_on="_date_dt",
        direction="nearest",
        tolerance=pd.Timedelta(days=int(window_days)),
    )
    merged = merged.drop(columns=["_anchor_dt", "_date_dt"], errors="ignore")
    return merged


def _safe_pct(actual: pd.Series, est: pd.Series) -> pd.Series:
    est2 = est.replace(0, pd.NA).astype("float64")
    return (actual.astype("float64") - est2) / est2.abs()


def save_artifacts(out_dir: Path, stem: str, df: pd.DataFrame, raw: Any, meta: Dict[str, Any]) -> None:
    ensure_dir(out_dir)
    raw_path = out_dir / f"{stem}.raw.json"
    csv_path = out_dir / f"{stem}.csv"
    meta_path = out_dir / f"{stem}.meta.json"

    write_json(raw, raw_path)
    df.to_csv(csv_path, index=False)
    write_json(meta, meta_path)

    print(f"[OK] {csv_path}  rows={len(df):,}")
    print(f"[OK] {raw_path}")
    print(f"[OK] {meta_path}")


def fetch_all_analyst_estimates(
    symbol: str,
    api_key: str,
    period: str,
    limit: int,
    sleep_s: float,
    max_pages: int = 200,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for page in range(max_pages):
        params = {"symbol": symbol, "period": period, "page": page, "limit": limit, "apikey": api_key}
        chunk = fetch_stable_endpoint("analyst-estimates", params=params, sleep_s=sleep_s)

        if not isinstance(chunk, list) or len(chunk) == 0:
            break

        for r in chunk:
            if isinstance(r, dict):
                all_rows.append(r)

        if len(chunk) < limit:
            break

    return all_rows


def _calendar_cutoff_from_data_dir(data_dir: Path, ticker: str) -> Optional[str]:
    """
    Return max(earnings_date) from:
      {data_dir}/{TICKER}/calendar/earnings_calendar.csv
    If not available, returns None.
    """
    cal_path = data_dir / ticker / "calendar" / "earnings_calendar.csv"
    if not cal_path.exists():
        return None
    try:
        df = pd.read_csv(cal_path)
    except Exception:
        return None

    if df.empty:
        return None

    col = "earnings_date" if "earnings_date" in df.columns else ("date" if "date" in df.columns else None)
    if col is None:
        return None

    dt = pd.to_datetime(df[col], errors="coerce")
    dt = dt.dropna()
    if dt.empty:
        return None

    return dt.max().strftime("%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download FMP STABLE income-statement / income-statement-growth / analyst-estimates and write tail-N aligned CSVs + surprises."
    )
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--sleep", type=float, default=0.25)

    ap.add_argument("--est-period", choices=["quarter", "annual"], default="quarter")
    ap.add_argument("--est-limit", type=int, default=100)

    ap.add_argument("--stmt-period", choices=["quarter", "annual"], default="quarter")
    ap.add_argument("--stmt-limit", type=int, default=400)

    # tail & alignment controls
    ap.add_argument("--tail-n", type=int, default=20, help="Keep last N statement dates (anchor).")
    ap.add_argument("--cutoff-end", default=None, help="Filter to date <= cutoff (YYYY-MM-DD).")
    ap.add_argument("--cutoff-from-calendar", action="store_true",
                    help="If set, use max(earnings_date) from data/{TICKER}/calendar/earnings_calendar.csv as cutoff-end.")
    ap.add_argument("--align-window-days", type=int, default=14, help="Nearest-match window for analyst estimates.")

    args = ap.parse_args()

    cfg = FMPConfig.from_env()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    for t in args.tickers:
        ticker = t.strip().upper()
        out_dir = data_dir / ticker / "financials"
        ensure_dir(out_dir)

        downloaded_at = _now_et_iso()

        # cutoff selection
        cutoff_end_used = args.cutoff_end
        cutoff_source = "arg"
        if args.cutoff_from_calendar:
            cal_cut = _calendar_cutoff_from_data_dir(data_dir, ticker)
            if cal_cut:
                cutoff_end_used = cal_cut
                cutoff_source = "earnings_calendar.csv"
            else:
                cutoff_source = "earnings_calendar.csv_missing_fallback_to_arg"

        # ---- (1) income-statement (actuals) ----
        params_stmt = {
            "symbol": ticker,
            "period": args.stmt_period,
            "limit": args.stmt_limit,
            "apikey": cfg.api_key,
        }
        raw_stmt = fetch_stable_endpoint("income-statement", params=params_stmt, sleep_s=args.sleep)
        df_stmt = _filter_cutoff(_prep_date(_to_df(raw_stmt)), cutoff_end_used)

        # anchor = statement dates
        anchor_dates = _tail_anchor_dates(df_stmt, int(args.tail_n))
        df_stmt_aligned = _align_exact(df_stmt, anchor_dates)

        meta_stmt = {
            "ticker": ticker,
            "endpoint": "stable/income-statement",
            "params": {k: v for k, v in params_stmt.items() if k != "apikey"},
            "downloaded_at_et": downloaded_at,
            "cutoff_end_used": cutoff_end_used,
            "cutoff_source": cutoff_source,
            "tail_n": int(args.tail_n),
            "rows_saved": int(df_stmt_aligned.shape[0]),
            "anchor_source": "income-statement",
            "sort": "date_asc",
        }
        save_artifacts(out_dir, f"stable_income_statement_{args.stmt_period}", df_stmt_aligned, raw_stmt, meta_stmt)

        # ---- (2) income-statement-growth ----
        raw_g = fetch_stable_endpoint("income-statement-growth", params=params_stmt, sleep_s=args.sleep)
        df_g = _filter_cutoff(_prep_date(_to_df(raw_g)), cutoff_end_used)
        df_g_aligned = _align_exact(df_g, anchor_dates)

        meta_g = {
            "ticker": ticker,
            "endpoint": "stable/income-statement-growth",
            "params": {k: v for k, v in params_stmt.items() if k != "apikey"},
            "downloaded_at_et": downloaded_at,
            "cutoff_end_used": cutoff_end_used,
            "cutoff_source": cutoff_source,
            "tail_n": int(args.tail_n),
            "rows_saved": int(df_g_aligned.shape[0]),
            "anchor_source": "income-statement",
            "sort": "date_asc",
        }
        save_artifacts(out_dir, f"stable_income_statement_growth_{args.stmt_period}", df_g_aligned, raw_g, meta_g)

        # ---- (3) analyst-estimates (all pages) ----
        rows_est = fetch_all_analyst_estimates(
            symbol=ticker,
            api_key=cfg.api_key,
            period=args.est_period,
            limit=args.est_limit,
            sleep_s=args.sleep,
        )
        df_est = _filter_cutoff(_prep_date(pd.DataFrame(rows_est)), cutoff_end_used)
        df_est_aligned = _align_nearest(df_est, anchor_dates, window_days=int(args.align_window_days))

        meta_est = {
            "ticker": ticker,
            "endpoint": "stable/analyst-estimates",
            "params": {"period": args.est_period, "limit": args.est_limit, "page": "looped_until_empty"},
            "downloaded_at_et": downloaded_at,
            "cutoff_end_used": cutoff_end_used,
            "cutoff_source": cutoff_source,
            "tail_n": int(args.tail_n),
            "align_window_days": int(args.align_window_days),
            "rows_saved": int(df_est_aligned.shape[0]),
            "anchor_source": "income-statement",
            "sort": "date_asc",
        }
        save_artifacts(out_dir, f"stable_analyst_estimates_{args.est_period}", df_est_aligned, rows_est, meta_est)

        # ---- (4) surprises (actual vs avg estimate) ----
        s = df_stmt_aligned.copy()
        e = df_est_aligned.copy()

        panel = pd.DataFrame({"period_end": s.get("date", pd.Series(dtype=str))})
        panel["quarter_rank"] = range(1, len(panel) + 1)

        for col in ["revenue", "ebitda", "ebit", "netIncome", "eps"]:
            if col in s.columns:
                panel[f"actual_{col}"] = pd.to_numeric(s[col], errors="coerce")

        est_map = {
            "revenueAvg": "revenue",
            "ebitdaAvg": "ebitda",
            "ebitAvg": "ebit",
            "netIncomeAvg": "netIncome",
            "epsAvg": "eps",
        }
        for src, dst in est_map.items():
            if src in e.columns:
                panel[f"est_{dst}"] = pd.to_numeric(e[src], errors="coerce")

        for m in ["revenue", "ebitda", "ebit", "netIncome", "eps"]:
            a = panel.get(f"actual_{m}")
            x = panel.get(f"est_{m}")
            if a is not None and x is not None:
                panel[f"surprise_{m}"] = a - x
                panel[f"surprise_{m}_pct"] = 100.0 * _safe_pct(a, x)

        stem = f"stable_surprises_{args.est_period}.tail{int(args.tail_n)}"
        meta_sur = {
            "ticker": ticker,
            "created_at_et": downloaded_at,
            "cutoff_end_used": cutoff_end_used,
            "cutoff_source": cutoff_source,
            "tail_n": int(args.tail_n),
            "anchor_source": "income-statement",
            "notes": "Row-aligned to income-statement anchor dates; surprise = actual - est_avg; pct uses /abs(est).",
        }
        save_artifacts(out_dir, stem, panel, raw={"derived_from": ["income-statement", "analyst-estimates"]}, meta=meta_sur)

        panel.to_csv(out_dir / f"stable_surprises_{args.est_period}.csv", index=False)

        print(f"[DONE] {ticker}\n")


if __name__ == "__main__":
    main()