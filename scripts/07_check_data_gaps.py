#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/07_check_data_gaps.py

One-line-per-ticker health check focused on:
1) calendar missing cells: count + where (printed)
2) grahamNumber missing: count only (printed, never a failure by itself)
3) transcript date non-alignment: summarize from gap vec (nz count + avg abs diff)
4) IMPORTANT: flag rows_total != expected (this catches MU 21 vs 20)

All detail is written to CSV.

Usage:
  python scripts/07_check_data_gaps.py --data-root data --expected 20 --out outputs/07_data_gap_report.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set

import pandas as pd


DATE_COL_CANDIDATES = [
    "date", "earningsDate", "earningDate", "earnings_date", "earning_date",
    "reportedDate", "announcementDate", "earningsDateTime", "datetime",
    "fiscalDateEnding", "fiscal_date_ending", "periodEndDate", "period_end_date",
]


def _log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}")


def _read_csv_pair(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not path.exists():
        return None, None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        _log("ERR", f"Failed to read CSV: {path} ({e})")
        return None, None

    try:
        df_str = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        df_str = df.astype(str).fillna("")
    return df, df_str


def _pick_date_col(df: pd.DataFrame) -> Optional[str]:
    cols = set(df.columns)
    for c in DATE_COL_CANDIDATES:
        if c in cols:
            return c
    for c in df.columns:
        if "date" in c.lower():
            return c
    return None


def _sort_and_tail(df: pd.DataFrame, df_str: pd.DataFrame, expected: int) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    date_col = _pick_date_col(df)
    if date_col is None:
        return df.tail(expected).copy(), df_str.tail(expected).copy(), None

    dts = pd.to_datetime(df[date_col], errors="coerce")
    tmp = df.copy()
    tmp_str = df_str.copy()
    tmp["_dt__"] = dts
    tmp_str["_dt__"] = dts
    tmp = tmp.sort_values("_dt__", kind="mergesort")
    tmp_str = tmp_str.sort_values("_dt__", kind="mergesort")
    out = tmp.tail(expected).drop(columns=["_dt__"])
    out_str = tmp_str.tail(expected).drop(columns=["_dt__"])
    return out.reset_index(drop=True), out_str.reset_index(drop=True), date_col


def _empty_mask(df: pd.DataFrame, df_str: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Vectorized empty detection:
      - NaN in df
      - '' after strip in df_str
      - literal 'nan' strings
    """
    df2 = df[cols]
    s2 = df_str[cols]

    na = df2.isna()
    stripped = s2.apply(lambda ser: ser.astype(str).str.strip())
    blanks = stripped.eq("")
    nan_str = stripped.apply(lambda ser: ser.str.lower().eq("nan"))
    return na | blanks | nan_str


def _fmt_date(x) -> str:
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return str(x)
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _summarize_calendar_where(df_lastN: pd.DataFrame, mask: pd.DataFrame, date_col: Optional[str], max_items: int = 6) -> str:
    items: List[str] = []
    if mask.empty:
        return ""
    bad_rows = mask.any(axis=1).to_numpy().nonzero()[0].tolist()
    for i in bad_rows[:max_items]:
        missing_cols = [c for c in mask.columns if bool(mask.loc[i, c])]
        if date_col and date_col in df_lastN.columns:
            d = _fmt_date(df_lastN.loc[i, date_col])
        else:
            d = f"row={i}"
        items.append(f"{d}:{','.join(missing_cols[:6])}")
    if len(bad_rows) > max_items:
        items.append(f"+{len(bad_rows) - max_items}more")
    return " | ".join(items)


@dataclass
class CsvCheckResult:
    exists: bool
    rows_total: int
    rows_lastN: int
    empty_cells_lastN: int
    empty_cols_lastN: List[str]
    date_col: Optional[str]
    where_compact: str


def _check_csv(path: Path, expected: int, cols_focus: Optional[Set[str]] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], CsvCheckResult]:
    df, df_str = _read_csv_pair(path)
    if df is None or df_str is None:
        return None, None, CsvCheckResult(False, 0, 0, 0, [], None, "")

    rows_total = int(len(df))
    lastN, lastN_str, date_col = _sort_and_tail(df, df_str, expected)
    cols = list(lastN.columns)

    if cols_focus is not None:
        cols = [c for c in cols if c in cols_focus]
        if not cols:
            return lastN, lastN_str, CsvCheckResult(True, rows_total, int(len(lastN)), 0, [], date_col, "")

    mask = _empty_mask(lastN, lastN_str, cols)
    empty_cells = int(mask.values.sum())
    empty_cols = [c for c in mask.columns if int(mask[c].sum()) > 0]
    where = _summarize_calendar_where(lastN, mask, date_col) if cols_focus is None else ""

    return lastN, lastN_str, CsvCheckResult(True, rows_total, int(len(lastN)), empty_cells, empty_cols, date_col, where)


def _calendar_lastN_dates_sorted(path: Path, expected: int) -> List[pd.Timestamp]:
    df, _ = _read_csv_pair(path)
    if df is None or df.empty:
        return []
    date_col = _pick_date_col(df)
    if date_col is None:
        return []

    dts = pd.to_datetime(df[date_col], errors="coerce")
    tmp = df.copy()
    tmp["_dt__"] = dts
    tmp = tmp.sort_values("_dt__", kind="mergesort")
    lastN = tmp.tail(expected)

    out: List[pd.Timestamp] = []
    for v in pd.to_datetime(lastN[date_col], errors="coerce"):
        if pd.isna(v):
            continue
        out.append(pd.Timestamp(v).normalize())
    out.sort()
    return out


def _transcripts_dates_lastN(tdir: Path, expected: int) -> Tuple[int, int, List[pd.Timestamp]]:
    subdirs = sorted([p for p in tdir.iterdir() if p.is_dir()])
    folders_total = len(subdirs)
    subdirs_lastN = subdirs[-expected:] if folders_total >= expected else subdirs

    dates: List[pd.Timestamp] = []
    for d in subdirs_lastN:
        ts = pd.to_datetime(d.name, errors="coerce")
        if pd.notna(ts):
            dates.append(pd.Timestamp(ts).normalize())
    dates.sort()
    return folders_total, len(subdirs_lastN), dates


def _date_gap_vector(cal_dates: List[pd.Timestamp], tx_dates: List[pd.Timestamp]) -> Optional[List[int]]:
    if len(cal_dates) != len(tx_dates):
        return None
    return [int(abs((c - t).days)) for c, t in zip(cal_dates, tx_dates)]


def _gap_summary(gaps: Optional[List[int]]) -> Tuple[str, int, int]:
    if gaps is None or len(gaps) == 0:
        return "NA", 0, 0
    nz = [g for g in gaps if g != 0]
    if not nz:
        return "nz=0 avg=0", 0, 0
    avg = sum(nz) / len(nz)
    avg_r = int(math.floor(avg + 0.5))
    return f"nz={len(nz)} avg={avg_r}", len(nz), avg_r


def _count_graham_missing_in_lastN(km_path: Path, expected: int) -> Tuple[int, str]:
    df, df_str = _read_csv_pair(km_path)
    if df is None or df_str is None or df.empty:
        return 0, ""
    lastN, lastN_str, date_col = _sort_and_tail(df, df_str, expected)
    if "grahamNumber" not in lastN.columns:
        return 0, ""

    mask = _empty_mask(lastN, lastN_str, ["grahamNumber"])
    missing_rows = mask["grahamNumber"].to_numpy().nonzero()[0].tolist()
    if not missing_rows:
        return 0, ""

    dates = []
    for i in missing_rows[:6]:
        if date_col and date_col in lastN.columns:
            dates.append(_fmt_date(lastN.loc[i, date_col]))
        else:
            dates.append(str(i))
    extra = f"+{len(missing_rows)-6}more" if len(missing_rows) > 6 else ""
    where = ",".join(dates) + (f" {extra}" if extra else "")
    return len(missing_rows), where


def _ticker_dirs(data_root: Path, tickers: Optional[List[str]]) -> List[Path]:
    if tickers:
        return [data_root / t for t in tickers if (data_root / t).is_dir()]
    out = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and (p / "calendar" / "earnings_calendar.csv").exists():
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--expected", type=int, default=20)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--out", type=str, default="outputs/07_data_gap_report.csv")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if real failures exist.")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_path = Path(args.out).resolve()
    expected = int(args.expected)

    if not data_root.exists():
        _log("ERR", f"data-root not found: {data_root}")
        return 2

    tick_dirs = _ticker_dirs(data_root, args.tickers)
    if not tick_dirs:
        _log("ERR", f"No tickers found under {data_root} (looking for calendar/earnings_calendar.csv).")
        return 2

    _log("INFO", f"Scanning {len(tick_dirs)} tickers under: {data_root}")

    rows: List[Dict[str, object]] = []
    any_real_fail = False

    for tdir in tick_dirs:
        ticker = tdir.name

        cal_path = tdir / "calendar" / "earnings_calendar.csv"
        km_path = tdir / "financials" / "key_metrics_quarter.csv"
        ra_path = tdir / "financials" / "ratios_quarter.csv"
        tdir_tx = tdir / "transcripts"

        # calendar (all cols)
        cal_lastN, cal_lastN_str, cal_chk = _check_csv(cal_path, expected, cols_focus=None)

        # key metrics / ratios basic check
        _, _, km_chk = _check_csv(km_path, expected, cols_focus=set())
        _, _, ra_chk = _check_csv(ra_path, expected, cols_focus=set())

        # graham missing only
        graham_missing, graham_where = _count_graham_missing_in_lastN(km_path, expected)

        # transcripts
        tx_exists = tdir_tx.exists()
        tx_folders_total = 0
        tx_folders_lastN = 0
        tx_missing_txt = 0
        tx_empty_txt = 0
        gap_vec: Optional[List[int]] = None
        gap_summary_str = "NA"

        if tx_exists:
            subdirs = sorted([p for p in tdir_tx.iterdir() if p.is_dir()])
            tx_folders_total = len(subdirs)
            subdirs_lastN = subdirs[-expected:] if tx_folders_total >= expected else subdirs
            tx_folders_lastN = len(subdirs_lastN)

            for d in subdirs_lastN:
                txt = d / "transcript.txt"
                if not txt.exists():
                    tx_missing_txt += 1
                else:
                    try:
                        if txt.stat().st_size == 0:
                            tx_empty_txt += 1
                    except Exception:
                        tx_empty_txt += 1

            cal_dates = _calendar_lastN_dates_sorted(cal_path, expected) if cal_chk.exists else []
            _, _, tx_dates = _transcripts_dates_lastN(tdir_tx, expected)
            gap_vec = _date_gap_vector(cal_dates, tx_dates)
            gap_summary_str, _, _ = _gap_summary(gap_vec)

        # failures
        real_fails: List[str] = []
        notes: List[str] = []

        if not cal_chk.exists:
            real_fails.append("missing_calendar")
        else:
            if cal_chk.rows_total != expected:
                real_fails.append(f"calendar_rows!={expected}")
            if cal_chk.empty_cells_lastN > 0:
                real_fails.append("calendar_missing_cells")

        if not km_chk.exists:
            real_fails.append("missing_key_metrics")
        else:
            if km_chk.rows_total != expected:
                real_fails.append(f"keymetrics_rows!={expected}")

        if not ra_chk.exists:
            real_fails.append("missing_ratios")
        else:
            if ra_chk.rows_total != expected:
                real_fails.append(f"ratios_rows!={expected}")

        if not tx_exists:
            real_fails.append("missing_transcripts_dir")
        else:
            if tx_folders_total != expected:
                real_fails.append(f"transcripts_count!={expected}")
            if tx_missing_txt > 0:
                real_fails.append("transcripts_missing_txt")
            if tx_empty_txt > 0:
                real_fails.append("transcripts_empty_txt")

        # notes
        if graham_missing > 0:
            notes.append("graham_missing")
        if gap_vec is None:
            notes.append("tx_gap_na")
        else:
            if any(g != 0 for g in gap_vec):
                notes.append("tx_gap_nonzero")

        status = "OK" if not real_fails else "WARN"
        if real_fails:
            any_real_fail = True

        cal_missing_n = cal_chk.empty_cells_lastN if cal_chk.exists else -1
        cal_cols = ",".join(cal_chk.empty_cols_lastN[:6]) if cal_chk.empty_cols_lastN else ""
        cal_where = cal_chk.where_compact if cal_missing_n > 0 else ""

        where_part = f" where={cal_where}" if cal_missing_n > 0 else ""

        _log(
            status,
            f"{ticker} | cal_rows={cal_chk.rows_total if cal_chk.exists else 0} miss={max(cal_missing_n,0)}"
            f"{(' cols=' + cal_cols) if cal_missing_n>0 and cal_cols else ''}"
            f"{where_part}"
            f" | graham_missing={graham_missing}"
            f" | tx_rows={tx_folders_total if tx_exists else 0} gap({gap_summary_str})"
            f"{(' | FAIL=' + ';'.join(real_fails)) if real_fails else ''}"
        )

        rows.append({
            "ticker": ticker,

            "calendar_exists": cal_chk.exists,
            "calendar_rows_total": cal_chk.rows_total,
            "calendar_rows_lastN": cal_chk.rows_lastN,
            "calendar_empty_cells_lastN": cal_chk.empty_cells_lastN,
            "calendar_empty_cols_lastN": ",".join(cal_chk.empty_cols_lastN),
            "calendar_missing_where_compact": cal_chk.where_compact,
            "calendar_date_col": cal_chk.date_col or "",

            "key_metrics_exists": km_chk.exists,
            "key_metrics_rows_total": km_chk.rows_total,
            "graham_missing_lastN": graham_missing,
            "graham_missing_where_compact": graham_where,

            "ratios_exists": ra_chk.exists,
            "ratios_rows_total": ra_chk.rows_total,

            "transcripts_exists": tx_exists,
            "transcripts_folders_total": tx_folders_total,
            "transcripts_folders_lastN": tx_folders_lastN,
            "transcripts_missing_transcript_txt": tx_missing_txt,
            "transcripts_empty_transcript_txt": tx_empty_txt,
            "transcripts_date_gap_vec_lastN": "" if gap_vec is None else str(gap_vec),
            "transcripts_gap_summary": gap_summary_str,

            "notes": ";".join(notes),
            "fail_reasons": ";".join(real_fails),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows).sort_values("ticker")
    df_out.to_csv(out_path, index=False)
    _log("INFO", f"Wrote report: {out_path}")

    if args.strict and any_real_fail:
        _log("ERR", "Strict mode: real failures detected.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
