#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timedelta, date
from pathlib import Path
import pandas as pd

from _common import repo_root, default_data_dir


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fmt_ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--data-dir", default=None)

    # News fetch controls
    ap.add_argument("--overwrite-news", action="store_true", help="Re-download news even if file exists")
    ap.add_argument("--pad-days", type=int, default=1, help="Pass-through to 05_fmp_news.py (internal query padding)")
    ap.add_argument("--page-limit", type=int, default=100, help="Pass-through to 05_fmp_news.py")
    ap.add_argument("--max-pages", type=int, default=200, help="Pass-through to 05_fmp_news.py")
    ap.add_argument("--sleep", type=float, default=0.2, help="Pass-through to 05_fmp_news.py")

    # Optional: build panel at end
    ap.add_argument("--build-panel", action="store_true", help="Run 10_build_event_panel_pilot.py at the end")
    ap.add_argument("--model-dir", default=None, help="Optional: pass to 10_build_event_panel_pilot.py")

    args = ap.parse_args()

    ticker = args.ticker.upper()
    start = args.start
    end = args.end

    root = repo_root()
    scripts = root / "scripts"

    # Resolve data dir (used to locate calendar + check news files)
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    base = ["python", "-u"]
    common: list[str] = []
    if args.data_dir:
        common += ["--data-dir", args.data_dir]

    # 0X: core data
    run(base + [str(scripts / "00_init_ticker_dirs.py"), "--ticker", ticker] + common)
    run(base + [str(scripts / "01_yf_prices.py"), "--ticker", ticker, "--start", start, "--end", end] + common)
    run(base + [str(scripts / "02_technicals.py"), "--ticker", ticker] + common)
    run(base + [str(scripts / "03_fmp_earnings_calendar.py"), "--ticker", ticker, "--start", start, "--end", end] + common)
    run(base + [str(scripts / "04_fmp_transcripts.py"), "--ticker", ticker, "--start", start, "--end", end] + common)

    # Read earnings dates to determine required news pull days (t-1, t+1)
    cal_path = data_dir / ticker / "calendar" / "earnings_calendar.csv"
    if not cal_path.exists():
        raise RuntimeError(f"Missing calendar file after 03: {cal_path}")

    cal = pd.read_csv(cal_path)
    if "earnings_date" not in cal.columns:
        raise RuntimeError(f"earnings_calendar.csv missing 'earnings_date': {cal_path}")

    cal["earnings_date"] = pd.to_datetime(cal["earnings_date"], errors="coerce").dt.date
    cal = cal.dropna(subset=["earnings_date"]).sort_values("earnings_date").reset_index(drop=True)

    # Only events inside [start,end] (calendar)
    start_d = parse_ymd(start)
    end_d = parse_ymd(end)
    cal = cal[(cal["earnings_date"] >= start_d) & (cal["earnings_date"] <= end_d)].copy()

    earnings_dates = cal["earnings_date"].tolist()
    if not earnings_dates:
        print(f"[WARN] No earnings dates found in {start}..{end} for {ticker}.")
        return

    # Required news dates: t-1 and t+1 for each earnings date
    news_dates = set()
    for t in earnings_dates:
        news_dates.add(t - timedelta(days=1))
        news_dates.add(t + timedelta(days=1))

    news_dates = sorted(news_dates)

    news_dir = data_dir / ticker / "news"
    news_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] {ticker}: {len(earnings_dates)} earnings events in range.")
    print(f"[INFO] {ticker}: will fetch news for {len(news_dates)} calendar days (t-1 and t+1 per event).")

    # 05: fetch news for each required day
    for d in news_dates:
        out_jsonl = news_dir / f"news_{fmt_ymd(d)}.jsonl"
        if out_jsonl.exists() and not args.overwrite_news:
            print(f"[SKIP] news exists: {out_jsonl.name}")
            continue

        cmd = (
            base
            + [str(scripts / "05_fmp_news.py"), "--ticker", ticker, "--date", fmt_ymd(d)]
            + common
            + [
                "--pad-days", str(args.pad_days),
                "--page-limit", str(args.page_limit),
                "--max-pages", str(args.max_pages),
                "--sleep", str(args.sleep),
            ]
        )

        try:
            run(cmd)
        except subprocess.CalledProcessError as e:
            # Don’t kill the whole run—log and continue so you still get most days.
            print(f"[WARN] news fetch failed for {ticker} {fmt_ymd(d)}: {e}")

    print("[OK] Basic ticker prep + news days done.")
    print(f"Next: build pilot panel with:\n  python scripts/10_build_event_panel_pilot.py --ticker {ticker}")

    # Optional: run panel build now
    if args.build_panel:
        panel_cmd = base + [str(scripts / "10_build_event_panel_pilot.py"), "--ticker", ticker] + common
        if args.model_dir:
            panel_cmd += ["--model-dir", args.model_dir]
        run(panel_cmd)
        print("[OK] Built pilot event panel.")


if __name__ == "__main__":
    main()
