# scripts/10_ticker_prep_all.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Tuple

from _common import DEFAULT_20


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        raise RuntimeError(f"Command failed (exit={r.returncode}): {' '.join(cmd)}")


def read_tickers_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"{path} is empty.")
            ticker_col = next((c for c in reader.fieldnames if c.strip().lower() == "ticker"), None)
            if ticker_col is None:
                raise ValueError(f"{path} has no 'ticker' column.")
            out: list[str] = []
            for row in reader:
                t = (row.get(ticker_col) or "").strip()
                if t:
                    out.append(t)
            return out

    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def clean_tickers(tickers: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in tickers:
        tt = t.strip().upper()
        if tt and tt not in seen:
            out.append(tt)
            seen.add(tt)
    return out


def _parse_iso_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        # accepts YYYY-MM-DD
        return datetime.fromisoformat(s).date()
    except Exception:
        return None


def calendar_cutoff_and_count(cal_csv: Path, ticker: str) -> Tuple[Optional[str], int]:
    """
    Returns:
      (max_earnings_date_iso, n_rows_for_ticker)
    If csv missing/unreadable/empty => (None, 0)
    """
    if not cal_csv.exists():
        return None, 0

    try:
        with cal_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None, 0

            # choose date column
            cols = {c.strip(): c for c in reader.fieldnames}
            if "earnings_date" in cols:
                date_col = cols["earnings_date"]
            elif "date" in cols:
                date_col = cols["date"]
            else:
                return None, 0

            tkr_col = cols.get("ticker", None)

            max_d: Optional[date] = None
            n = 0
            for row in reader:
                if tkr_col:
                    rt = (row.get(tkr_col) or "").strip().upper()
                    if rt and rt != ticker:
                        continue

                d = _parse_iso_date(row.get(date_col, ""))
                if d is None:
                    continue
                n += 1
                if max_d is None or d > max_d:
                    max_d = d

            if max_d is None:
                return None, 0
            return max_d.isoformat(), n

    except Exception:
        return None, 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prep per-ticker data: init dirs, Yahoo prices+technicals, FMP calendar+transcripts+financials, and FMP STABLE statements/estimates."
    )

    # tickers
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--tickers-file", default=None)
    ap.add_argument("--max-tickers", type=int, default=None)

    # window + expected count (earnings events)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument(
        "--price-start",
        default="2021-01-01",
        help="Earliest analysis date for Yahoo Finance download (before buffers).",
    )
    ap.add_argument("--yf-buffer-before-months", type=int, default=15,
                    help="Download this many months before --price-start (default 15).")
    ap.add_argument("--yf-buffer-after-months", type=int, default=1,
                    help="Download this many months after --end (default 1).")
    ap.add_argument("--expected", type=int, default=20)

    # storage
    ap.add_argument("--data-dir", default="data")

    # throttling
    ap.add_argument("--sleep", type=float, default=0.25)

    # skips
    ap.add_argument("--skip_yf", action="store_true", help="Skip Yahoo prices + technicals.")
    ap.add_argument("--skip_technicals", action="store_true")
    ap.add_argument("--skip_fmp", action="store_true", help="Skip legacy FMP: calendar/transcripts/financials/news.")
    ap.add_argument("--skip_transcripts", action="store_true")
    ap.add_argument("--skip_financials", action="store_true")
    ap.add_argument("--with_news", action="store_true")
    ap.set_defaults(with_news=True)
    
    # News window (earnings-date centered): [-pre_bdays, +post_bdays], weekdays-only
    ap.add_argument("--news-pre-bdays", type=int, default=5)
    ap.add_argument("--news-post-bdays", type=int, default=10)
    ap.add_argument("--news-page-limit", type=int, default=100)
    ap.add_argument("--news-max-pages", type=int, default=100)
    ap.add_argument("--news-chunk-days", type=int, default=0, help="0 = no chunking; fetch window in one call")
    ap.add_argument("--news-pad-days", type=int, default=1)
    ap.add_argument("--skip_stable", action="store_true", help="Skip STABLE endpoints (script 08).")

    # stable controls
    ap.add_argument("--stable-tail-n", type=int, default=20)
    ap.add_argument("--stable-align-window-days", type=int, default=14)
    ap.add_argument("--stable-stmt-period", choices=["quarter", "annual"], default="quarter")
    ap.add_argument("--stable-est-period", choices=["quarter", "annual"], default="quarter")
    ap.add_argument("--stable-stmt-limit", type=int, default=400)
    ap.add_argument("--stable-est-limit", type=int, default=100)

    # NEW: stable cutoff derived from earnings_calendar.csv max date
    ap.add_argument(
        "--stable-cutoff-from-calendar",
        action="store_true",
        default=True,
        help="Use max(earnings_date) in data/{TICKER}/calendar/earnings_calendar.csv as cutoff for STABLE (fixes off-by-one). Default: True.",
    )

    # post checks
    ap.add_argument("--skip_gap_check", action="store_true")
    ap.add_argument("--gap_report", default="outputs/07_data_gap_report.csv")
    ap.add_argument("--continue_on_error", action="store_true")

    args = ap.parse_args()

    # ---- resolve tickers ----
    tickers: list[str] = []
    if args.ticker:
        tickers = [args.ticker]
    else:
        if args.tickers_file:
            tickers += read_tickers_file(Path(args.tickers_file))
        if args.tickers:
            tickers += args.tickers
        if not tickers:
            tickers = DEFAULT_20

    tickers = clean_tickers(tickers)
    if args.max_tickers is not None:
        tickers = tickers[: args.max_tickers]

    # ---- env requirement ----
    needs_fmp = (not args.skip_fmp) or (not args.skip_stable)
    if needs_fmp and not os.getenv("FMP_API_KEY"):
        raise RuntimeError("FMP_API_KEY not set. Example:\n  export FMP_API_KEY='...'\n")

    root = repo_root()
    scripts = root / "scripts"
    py = sys.executable

    print(
        f"[CONFIG] DATA_DIR={args.data_dir} range={args.start}..{args.end} tickers={tickers} "
        f"skip_yf={args.skip_yf} skip_fmp={args.skip_fmp} skip_transcripts={args.skip_transcripts} "
        f"skip_financials={args.skip_financials} skip_stable={args.skip_stable} with_news={args.with_news} "
        f"sleep={args.sleep} price_start={args.price_start} yf_buffer_before={args.yf_buffer_before_months} yf_buffer_after={args.yf_buffer_after_months}",
        flush=True,
    )

    common_data_dir = ["--data-dir", args.data_dir]

    for i, tkr in enumerate(tickers, start=1):
        tkr = tkr.upper()
        print(f"\n=== [{i}/{len(tickers)}] {tkr} ===", flush=True)

        try:
            # 00 init
            run([py, "-u", str(scripts / "00_init_ticker_dirs.py"), "--ticker", tkr] + common_data_dir, cwd=root)

            # 01 prices (uses --outdir)
            if not args.skip_yf:
                run(
                    [
                        py,
                        "-u",
                        str(scripts / "01_yf_prices.py"),
                        "--ticker",
                        tkr,
                        "--start",
                        args.price_start,
                        "--end",
                        args.end,
                        "--buffer-before-months",
                        str(int(args.yf_buffer_before_months)),
                        "--buffer-after-months",
                        str(int(args.yf_buffer_after_months)),
                        "--outdir",
                        args.data_dir,
                    ],
                    cwd=root,
                )
            # 02 technicals
            if not args.skip_yf and not args.skip_technicals:
                run([py, "-u", str(scripts / "02_technicals.py"), "--ticker", tkr] + common_data_dir, cwd=root)

            # Legacy FMP: 03/04/06/05
            cal_csv: Optional[Path] = None
            cal_cutoff: Optional[str] = None
            cal_n: int = 0

            if not args.skip_fmp:
                run(
                    [
                        py,
                        "-u",
                        str(scripts / "03_fmp_earnings_calendar.py"),
                        "--ticker",
                        tkr,
                        "--start",
                        args.start,
                        "--end",
                        args.end,
                        "--expected",
                        str(int(args.expected)),
                    ]
                    + common_data_dir,
                    cwd=root,
                )

                # locate calendar file + compute cutoff (max earnings_date)
                cal_csv = Path(args.data_dir) / tkr / "calendar" / "earnings_calendar.csv"
                cal_cutoff, cal_n = calendar_cutoff_and_count(cal_csv, tkr)

                if not args.skip_transcripts:
                    run(
                        [
                            py,
                            "-u",
                            str(scripts / "04_fmp_transcripts.py"),
                            "--ticker",
                            tkr,
                            "--start",
                            args.start,
                            "--end",
                            args.end,
                            "--expected",
                            str(int(args.expected)),
                        ]
                        + common_data_dir,
                        cwd=root,
                    )

                if not args.skip_financials:
                    run(
                        [
                            py,
                            "-u",
                            str(scripts / "06_fmp_financials.py"),
                            "--ticker",
                            tkr,
                            "--start",
                            args.start,
                            "--end",
                            args.end,
                            "--expected",
                            str(int(args.expected)),
                            "--limit",
                            "400",
                        ]
                        + common_data_dir,
                        cwd=root,
                    )

                if args.with_news:
                    # Pull news in an earnings-centered window for EACH earnings date in the calendar.
                    cal_csv = Path(args.data_dir) / tkr / "calendar" / "earnings_calendar.csv"
                    if not cal_csv.exists():
                        raise FileNotFoundError(
                            f"Missing earnings_calendar.csv for {tkr}. Expected: {cal_csv}. "
                            "Run without --skip_fmp (or ensure script 03 ran) before --with_news."
                        )

                    earnings_dates: list[str] = []
                    with cal_csv.open("r", newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        fields = [c.strip() for c in (reader.fieldnames or [])]
                        if "earnings_date" not in fields:
                            raise ValueError(
                                f"{cal_csv} missing 'earnings_date' column. Found: {fields}"
                            )
                        for row in reader:
                            d = (row.get("earnings_date") or "").strip()
                            if d:
                                earnings_dates.append(d)

                    # De-duplicate + stable order
                    earnings_dates = sorted(set(earnings_dates))

                    if not earnings_dates:
                        print(f"[WARN] {tkr}: no earnings dates found in {cal_csv}; skipping news")
                    else:
                        for ed in earnings_dates:
                            run(
                                [
                                    py,
                                    "-u",
                                    str(scripts / "05_fmp_news.py"),
                                    "--ticker",
                                    tkr,
                                    "--earnings-date",
                                    ed,
                                    "--pre-bdays",
                                    str(int(args.news_pre_bdays)),
                                    "--post-bdays",
                                    str(int(args.news_post_bdays)),
                                    "--page-limit",
                                    str(int(args.news_page_limit)),
                                    "--max-pages",
                                    str(int(args.news_max_pages)),
                                    "--chunk-days",
                                    str(int(args.news_chunk_days)),
                                    "--pad-days",
                                    str(int(args.news_pad_days)),
                                    "--sleep",
                                    str(float(args.sleep)),
                                ]
                                + common_data_dir,
                                cwd=root,
                            )

            # STABLE: 08 (tail-N aligned), cutoff from earnings_calendar max date
            if not args.skip_stable:
                stable_cutoff_end = args.end
                stable_tail_n = int(args.stable_tail_n)

                if args.stable_cutoff_from_calendar and cal_cutoff:
                    stable_cutoff_end = cal_cutoff
                    # keep row counts aligned with calendar if calendar is shorter
                    if cal_n > 0:
                        stable_tail_n = min(stable_tail_n, cal_n)

                run(
                    [
                        py,
                        "-u",
                        str(scripts / "08_fmp_stable_income_and_estimates.py"),
                        "--tickers",
                        tkr,
                        "--data-dir",
                        args.data_dir,
                        "--sleep",
                        str(float(args.sleep)),
                        "--stmt-period",
                        args.stable_stmt_period,
                        "--stmt-limit",
                        str(int(args.stable_stmt_limit)),
                        "--est-period",
                        args.stable_est_period,
                        "--est-limit",
                        str(int(args.stable_est_limit)),
                        "--tail-n",
                        str(int(stable_tail_n)),
                        "--cutoff-end",
                        stable_cutoff_end,
                        "--align-window-days",
                        str(int(args.stable_align_window_days)),
                    ],
                    cwd=root,
                )

            print(f"[OK] Done: {tkr}", flush=True)
            time.sleep(float(args.sleep))

        except Exception as e:
            print(f"[ERR] {tkr}: {e}", flush=True)
            if not args.continue_on_error:
                raise

    # Gap check once at end
    if not args.skip_gap_check:
        run(
            [
                py,
                "-u",
                str(scripts / "07_check_data_gaps.py"),
                "--data-root",
                args.data_dir,
                "--expected",
                str(int(args.expected)),
                "--out",
                args.gap_report,
                "--tickers",
                *tickers,
            ],
            cwd=root,
        )

    print("\n[OK] Prep complete.", flush=True)


if __name__ == "__main__":
    main()
