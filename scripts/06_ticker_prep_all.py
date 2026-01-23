#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def read_tickers_file(path: Path) -> list[str]:
    """
    Accepts either:
      - CSV with a 'ticker' column
      - or plain text file with one ticker per line (comments allowed with #)
    """
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"{path} is empty.")
            ticker_col = None
            for c in reader.fieldnames:
                if c.strip().lower() == "ticker":
                    ticker_col = c
                    break
            if ticker_col is None:
                raise ValueError(f"{path} has no 'ticker' column.")
            out = []
            for row in reader:
                t = (row.get(ticker_col) or "").strip()
                if t:
                    out.append(t)
            return out

    # txt-like
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def clean_tickers(tickers: list[str]) -> list[str]:
    out = []
    seen = set()
    for t in tickers:
        tt = t.strip().upper()
        if not tt:
            continue
        if tt not in seen:
            out.append(tt)
            seen.add(tt)
    return out


DEFAULT_20 = [
    "NVDA", "GOOGL", "AAPL", "MSFT", "AMZN",
    "META", "AVGO", "TSLA", "LLY", "WMT", 
    "JPM", "V", "XOM", "JNJ", "ORCL",
    "MA", "MU", "COST", "AMD", "PLTR",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare per-ticker data: YF prices + technicals + FMP earnings calendar + transcripts (news optional)."
    )

    # Backward compatible single ticker
    ap.add_argument("--ticker", default=None, help="Single ticker (legacy mode). Example: --ticker NVDA")

    # New batch interfaces
    ap.add_argument("--tickers", nargs="*", default=None, help="Space-separated tickers. Example: --tickers AAPL MSFT NVDA")
    ap.add_argument("--tickers-file", default=None, help="CSV(with ticker col) or txt file (one per line).")

    # Date range
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")

    # Storage
    ap.add_argument("--data-dir", default=None, help="Pass through to scripts if supported; default is repo_root/data.")

    # Controls
    ap.add_argument("--skip-technicals", action="store_true", help="Skip 02_technicals.py")
    ap.add_argument("--skip-fmp", action="store_true", help="Skip FMP steps (03 calendar + 04 transcripts).")
    ap.add_argument("--skip-transcripts", action="store_true", help="Skip 04_fmp_transcripts.py (still runs 03).")

    # News is optional and OFF by default
    ap.add_argument("--with-news", action="store_true", help="Also fetch news (not recommended for bulk). Off by default.")
    ap.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between major calls (rate-limit friendly).")

    ap.add_argument("--continue-on-error", action="store_true", help="Keep going if one ticker fails.")
    ap.add_argument("--max-tickers", type=int, default=None, help="Optional cap for quick testing.")

    args = ap.parse_args()

    # Resolve tickers with precedence:
    # 1) --ticker (single)
    # 2) --tickers-file / --tickers
    # 3) default list
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

    # FMP key requirement (only if using FMP)
    if not args.skip_fmp and not os.getenv("FMP_API_KEY"):
        raise RuntimeError(
            "FMP_API_KEY not set. Do: export FMP_API_KEY='...'\n"
            "(Or run with --skip-fmp)"
        )

    root = repo_root()
    scripts = root / "scripts"
    py = sys.executable  # ensures we use the active venv python

    common = []
    if args.data_dir:
        common += ["--data-dir", args.data_dir]

    print(f"[INFO] Repo root: {root}", flush=True)
    print(f"[INFO] Tickers ({len(tickers)}): {tickers}", flush=True)
    print(f"[INFO] Range: {args.start} .. {args.end}", flush=True)
    print(f"[INFO] skip_fmp={args.skip_fmp} skip_technicals={args.skip_technicals} skip_transcripts={args.skip_transcripts} with_news={args.with_news}", flush=True)

    for idx, tkr in enumerate(tickers, start=1):
        print(f"\n=== [{idx}/{len(tickers)}] {tkr} ===", flush=True)
        try:
            run([py, "-u", str(scripts / "00_init_ticker_dirs.py"), "--ticker", tkr] + common, cwd=root)

            run([py, "-u", str(scripts / "01_yf_prices.py"), "--ticker", tkr, "--start", args.start, "--end", args.end] + common, cwd=root)

            if not args.skip_technicals:
                run([py, "-u", str(scripts / "02_technicals.py"), "--ticker", tkr] + common, cwd=root)

            if not args.skip_fmp:
                run([py, "-u", str(scripts / "03_fmp_earnings_calendar.py"), "--ticker", tkr, "--start", args.start, "--end", args.end] + common, cwd=root)
                if not args.skip_transcripts:
                    run([py, "-u", str(scripts / "04_fmp_transcripts.py"), "--ticker", tkr, "--start", args.start, "--end", args.end] + common, cwd=root)

            if args.with_news and (not args.skip_fmp):
                # If you later want bulk news, we can add a controlled loop here.
                # For now, keep it explicit by running 05 per-day to avoid surprises.
                print(f"[WARN] --with-news currently does not bulk-fetch automatically. Use 05_fmp_news.py per date.", flush=True)

            print(f"[OK] Done: {tkr}", flush=True)
            time.sleep(args.sleep)

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {tkr} failed with exit code {e.returncode}", flush=True)
            if not args.continue_on_error:
                raise

    print("\n[OK] Prep complete.", flush=True)
    print("Next (optional): build event panel per ticker with:", flush=True)
    print("  python scripts/10_build_event_panel_pilot.py --ticker <TICKER>", flush=True)


if __name__ == "__main__":
    main()
