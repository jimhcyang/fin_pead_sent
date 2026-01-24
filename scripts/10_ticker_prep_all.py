#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, date
from pathlib import Path


DEFAULT_20 = [
    "NVDA", "GOOGL", "AAPL", "MSFT", "AMZN",
    "META", "AVGO", "TSLA", "LLY", "WMT",
    "JPM", "V", "XOM", "JNJ", "ORCL",
    "MA", "MU", "COST", "AMD", "PLTR",
]

DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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
        if not tt:
            continue
        if tt not in seen:
            out.append(tt)
            seen.add(tt)
    return out


def _parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _data_base(root: Path, data_dir: str | None) -> Path:
    if data_dir:
        p = Path(data_dir)
        return (p if p.is_absolute() else (root / p)).resolve()
    return (root / "data").resolve()


def _latest_transcript_date(transcripts_dir: Path) -> date:
    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Missing transcripts dir: {transcripts_dir}")
    dates: list[date] = []
    for child in transcripts_dir.iterdir():
        if child.is_dir() and DATE_DIR_RE.match(child.name):
            dates.append(_parse_yyyy_mm_dd(child.name))
    if not dates:
        raise RuntimeError(f"No transcript date folders found under: {transcripts_dir}")
    return max(dates)


def _last_csv_date(csv_path: Path, candidates: tuple[str, ...] = ("earnings_date", "date")) -> date:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {csv_path}")

        cols = [c.strip() for c in reader.fieldnames]
        col = next((c for c in candidates if c in cols), None)
        if col is None:
            raise ValueError(f"CSV has none of columns {candidates}: {csv_path} (cols={cols})")

        last: date | None = None
        for row in reader:
            ds = (row.get(col) or "").strip()
            if not ds:
                continue
            last = _parse_yyyy_mm_dd(ds)

    if last is None:
        raise ValueError(f"No valid dates found in column {col}: {csv_path}")
    return last


def _help_text(py: str, script_path: Path, cwd: Path) -> str:
    try:
        out = subprocess.check_output(
            [py, str(script_path), "-h"],
            cwd=str(cwd),
            text=True,
            stderr=subprocess.STDOUT,
        )
        return out or ""
    except Exception:
        return ""


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare per-ticker data: YF prices + technicals + FMP earnings calendar + transcripts + aligned financials."
    )

    # Single ticker (legacy)
    ap.add_argument("--ticker", default=None, help="Single ticker (legacy mode). Example: --ticker NVDA")

    # Batch
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

    # Financials
    ap.add_argument("--skip-financials", action="store_true", help="Skip 06_fmp_financials.py")
    ap.add_argument("--financials-limit", type=int, default=400, help="Row limit for financials endpoints.")

    # Keep these for future compatibility, but we only pass them if 06 supports them.
    # NOTE: your latest 06 script does NOT define --period / --with-statements, so these will be auto-ignored.
    ap.add_argument("--financials-period", choices=["quarter", "annual"], default="quarter", help="(Only used if 06 supports --period).")
    ap.add_argument("--with-statements", action="store_true", help="(Only used if 06 supports --with-statements).")
    ap.add_argument("--save-as-is", action="store_true", help="(Only used if 06 supports --save-as-is).")

    # News optional
    ap.add_argument("--with-news", action="store_true", help="Also fetch news (not recommended for bulk). Off by default.")
    ap.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between tickers.")

    ap.add_argument("--continue-on-error", action="store_true", help="Keep going if one ticker fails.")
    ap.add_argument("--max-tickers", type=int, default=None, help="Optional cap for quick testing.")

    # Sanity checks
    ap.add_argument(
        "--no-align-check",
        action="store_true",
        help="Disable alignment checks (meta matched/events + last transcript date == last row date).",
    )

    args = ap.parse_args()

    # Resolve tickers
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

    root = repo_root()
    scripts = root / "scripts"
    py = sys.executable
    data_base = _data_base(root, args.data_dir)

    common: list[str] = []
    if args.data_dir:
        common += ["--data-dir", args.data_dir]

    # If financials ON, FORCE transcripts/calendar (alignment depends on transcript events)
    force_fmp_for_financials = not args.skip_financials
    if force_fmp_for_financials and (args.skip_fmp or args.skip_transcripts):
        print(
            "[WARN] Financials alignment requires calendar+transcripts. "
            "Overriding --skip-fmp/--skip-transcripts for this run.",
            flush=True,
        )

    # Will we run any FMP step?
    will_run_fmp = (not args.skip_fmp) or force_fmp_for_financials

    if will_run_fmp and not os.getenv("FMP_API_KEY"):
        raise RuntimeError(
            "FMP_API_KEY not set. Do: export FMP_API_KEY='...'\n"
            "(Or run with --skip-fmp and --skip-financials)"
        )

    # Detect supported flags for 06 once
    fin_script = scripts / "06_fmp_financials.py"
    fin_help = _help_text(py, fin_script, root)
    fin_supports_period = "--period" in fin_help
    fin_supports_statements = "--with-statements" in fin_help
    fin_supports_save_as_is = "--save-as-is" in fin_help

    print(f"[INFO] Repo root: {root}", flush=True)
    print(f"[INFO] Data base: {data_base}", flush=True)
    print(f"[INFO] Tickers ({len(tickers)}): {tickers}", flush=True)
    print(f"[INFO] Range: {args.start} .. {args.end}", flush=True)
    print(
        f"[INFO] skip_technicals={args.skip_technicals} skip_fmp={args.skip_fmp} "
        f"skip_transcripts={args.skip_transcripts} skip_financials={args.skip_financials} "
        f"financials_limit={args.financials_limit} align_check={not args.no_align_check}",
        flush=True,
    )
    print(
        "[INFO] 06_fmp_financials.py supports:"
        f" --period={fin_supports_period}"
        f" --with-statements={fin_supports_statements}"
        f" --save-as-is={fin_supports_save_as_is}",
        flush=True,
    )

    for idx, tkr in enumerate(tickers, start=1):
        print(f"\n=== [{idx}/{len(tickers)}] {tkr} ===", flush=True)
        try:
            # 00 init
            run([py, "-u", str(scripts / "00_init_ticker_dirs.py"), "--ticker", tkr] + common, cwd=root)

            # 01 prices
            run(
                [py, "-u", str(scripts / "01_yf_prices.py"), "--ticker", tkr, "--start", args.start, "--end", args.end] + common,
                cwd=root,
            )

            # 02 technicals
            if not args.skip_technicals:
                run([py, "-u", str(scripts / "02_technicals.py"), "--ticker", tkr] + common, cwd=root)

            # 03 + 04 (forced if financials on)
            if will_run_fmp:
                run(
                    [py, "-u", str(scripts / "03_fmp_earnings_calendar.py"), "--ticker", tkr, "--start", args.start, "--end", args.end] + common,
                    cwd=root,
                )
                run(
                    [py, "-u", str(scripts / "04_fmp_transcripts.py"), "--ticker", tkr, "--start", args.start, "--end", args.end] + common,
                    cwd=root,
                )

            if args.with_news and will_run_fmp:
                print("[WARN] --with-news not bulk-fetched in this runner. Use 05_fmp_news.py separately.", flush=True)

            # 06 financials
            if not args.skip_financials:
                cmd = [
                    py, "-u", str(fin_script),
                    "--ticker", tkr,
                    "--start", args.start,
                    "--end", args.end,
                    "--limit", str(args.financials_limit),
                ] + common

                # Only pass optional flags if 06 supports them
                if args.financials_period != "quarter" and not fin_supports_period:
                    print(f"[WARN] 06 does not support --period; ignoring --financials-period {args.financials_period}", flush=True)
                if fin_supports_period:
                    cmd += ["--period", args.financials_period]

                if args.with_statements and not fin_supports_statements:
                    print("[WARN] 06 does not support --with-statements; ignoring --with-statements", flush=True)
                if args.with_statements and fin_supports_statements:
                    cmd += ["--with-statements"]

                if args.save_as_is and not fin_supports_save_as_is:
                    print("[WARN] 06 does not support --save-as-is; ignoring --save-as-is", flush=True)
                if args.save_as_is and fin_supports_save_as_is:
                    cmd += ["--save-as-is"]

                run(cmd, cwd=root)

                # Alignment checks (IMPROVED)
                if not args.no_align_check:
                    transcripts_dir = data_base / tkr / "transcripts"
                    fin_dir = data_base / tkr / "financials"

                    km_csv = fin_dir / "key_metrics_quarter.csv"
                    ra_csv = fin_dir / "ratios_quarter.csv"

                    km_meta_p = fin_dir / "key_metrics_quarter.meta.json"
                    ra_meta_p = fin_dir / "ratios_quarter.meta.json"

                    # 1) Meta-based check (catches "all-NaN" failures even if dates look OK)
                    km_meta = _read_json(km_meta_p)
                    ra_meta = _read_json(ra_meta_p)

                    km_events = int(km_meta.get("events", 0) or 0)
                    ra_events = int(ra_meta.get("events", 0) or 0)
                    km_matched = int(km_meta.get("matched", 0) or 0)
                    ra_matched = int(ra_meta.get("matched", 0) or 0)

                    if km_events <= 0 or ra_events <= 0:
                        raise RuntimeError(f"[ALIGNMENT FAIL] {tkr}: meta events missing/zero (km_events={km_events}, ra_events={ra_events}).")

                    if km_matched != km_events or ra_matched != ra_events:
                        raise RuntimeError(
                            f"[ALIGNMENT FAIL] {tkr}: matched != events "
                            f"(key_metrics matched={km_matched}/{km_events}, ratios matched={ra_matched}/{ra_events})."
                        )

                    if int(km_meta.get("missing_keys_in_events", 0) or 0) > 0 or int(ra_meta.get("missing_keys_in_events", 0) or 0) > 0:
                        raise RuntimeError(
                            f"[ALIGNMENT FAIL] {tkr}: missing fiscalYear/period in transcript events "
                            f"(km_missing_keys={km_meta.get('missing_keys_in_events')}, ra_missing_keys={ra_meta.get('missing_keys_in_events')})."
                        )

                    # 2) Date-based check (kept; useful sanity check)
                    t_last = _latest_transcript_date(transcripts_dir)
                    km_last = _last_csv_date(km_csv, candidates=("earnings_date", "date"))
                    ra_last = _last_csv_date(ra_csv, candidates=("earnings_date", "date"))

                    if km_last != t_last or ra_last != t_last:
                        raise RuntimeError(
                            f"[ALIGNMENT FAIL] {tkr}: last transcript date={t_last} "
                            f"but key_metrics last date={km_last}, ratios last date={ra_last}."
                        )

                    print(
                        f"[CHECK] Alignment OK: {tkr} events={km_events} matched={km_matched} last_date={t_last}",
                        flush=True,
                    )

            print(f"[OK] Done: {tkr}", flush=True)
            time.sleep(args.sleep)

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {tkr} failed with exit code {e.returncode}", flush=True)
            if not args.continue_on_error:
                raise
        except Exception as e:
            print(f"[ERROR] {tkr} failed: {e}", flush=True)
            if not args.continue_on_error:
                raise

    print("\n[OK] Prep complete.", flush=True)


if __name__ == "__main__":
    main()
