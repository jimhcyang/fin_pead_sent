#!/usr/bin/env python3
# scripts/30_build_events_all.py

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_20 = [
    "NVDA", "GOOGL", "AAPL", "MSFT", "AMZN",
    "META", "AVGO", "TSLA", "LLY", "WMT",
    "JPM", "V", "XOM", "JNJ", "ORCL",
    "MA", "MU", "COST", "AMD", "PLTR",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


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


def must_exist(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing script: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build per-ticker event panels (21–24), download SPX once (25), then compute abnormal returns (26)."
    )

    # tickers
    ap.add_argument("--ticker", default=None, help="Single ticker (optional).")
    ap.add_argument("--tickers", nargs="*", default=None, help="Space-separated tickers.")
    ap.add_argument("--tickers-file", default=None, help="CSV(with ticker col) or txt file (one per line).")

    # storage
    ap.add_argument("--data-dir", default=None, help="Override repo_root/data")

    # SPX download range (used by script 25)
    ap.add_argument("--start", default="2021-01-01", help="YYYY-MM-DD (for SPX download)")
    ap.add_argument("--end", default="2025-12-31", help="YYYY-MM-DD (for SPX download)")
    ap.add_argument("--spx-symbol", default="^GSPC", help="Yahoo symbol for market proxy (default ^GSPC)")
    ap.add_argument("--spx-rel", default="_tmp_market/spx/prices/yf_ohlcv_daily.csv",
                    help="Relative path under data/ where SPX is stored (must match 26 --spx-rel).")
    ap.add_argument("--force-spx", action="store_true", help="Force redownload SPX even if file exists.")

    # script-26 wiring
    ap.add_argument("--panel-name", default="event_panel_numeric", help="Input panel name for 26 (from script 24 output).")
    ap.add_argument("--out-name", default="event_panel", help="Output name for 26.")

    # controls
    ap.add_argument("--sleep", type=float, default=0.10)
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--max-tickers", type=int, default=None)

    args = ap.parse_args()

    root = repo_root()
    scripts = root / "scripts"
    py = sys.executable

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

    common: list[str] = []
    if args.data_dir:
        common += ["--data-dir", args.data_dir]

    # ---- script paths (EXPECTED names) ----
    s21 = scripts / "21_build_event_anchors.py"
    s22 = scripts / "22_extract_event_price_path.py"
    s23 = scripts / "23_compute_event_returns.py"
    s24 = scripts / "24_merge_event_fundamentals.py"
    s25 = scripts / "25_yf_download_spx.py"

    s26 = scripts / "26_compute_abnormal_returns.py"

    # fail fast if scripts missing
    for sp in (s21, s22, s23, s24, s25, s26):
        must_exist(sp)

    print(f"[INFO] Repo root: {root}", flush=True)
    print(f"[INFO] Tickers ({len(tickers)}): {tickers}", flush=True)
    print(f"[INFO] SPX: symbol={args.spx_symbol} range={args.start}..{args.end} rel={args.spx_rel}", flush=True)

    # ---- 25: download SPX ONCE ----
    cmd25 = [
        py, "-u", str(s25),
        "--symbol", args.spx_symbol,
        "--start", args.start,
        "--end", args.end,
        "--out-rel", args.spx_rel,
    ] + common
    if args.force_spx:
        cmd25 += ["--force"]
    run(cmd25, cwd=root)

    # ---- 21–24 per ticker, then 26 per ticker ----
    for i, tkr in enumerate(tickers, start=1):
        tkr = tkr.upper()
        print(f"\n=== [{i}/{len(tickers)}] {tkr} ===", flush=True)
        try:
            run([py, "-u", str(s21), "--ticker", tkr] + common, cwd=root)
            run([py, "-u", str(s22), "--ticker", tkr] + common, cwd=root)
            run([py, "-u", str(s23), "--ticker", tkr] + common, cwd=root)
            run([py, "-u", str(s24), "--ticker", tkr] + common, cwd=root)

            run(
                [
                    py, "-u", str(s26),
                    "--ticker", tkr,
                    "--panel-name", args.panel_name,
                    "--spx-rel", args.spx_rel,
                    "--out-name", args.out_name,
                ] + common,
                cwd=root,
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

    print("\n[OK] Event build complete (21–26).", flush=True)


if __name__ == "__main__":
    main()
