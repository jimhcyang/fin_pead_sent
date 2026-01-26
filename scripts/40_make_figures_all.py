#!/usr/bin/env python3
# scripts/40_make_figures_all.py

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


DEFAULT_20 = [
    "NVDA", "GOOGL", "AAPL", "MSFT", "AMZN",
    "META", "AVGO", "TSLA", "LLY", "WMT",
    "JPM", "V", "XOM", "JNJ", "ORCL",
    "MA", "MU", "COST", "AMD", "PLTR",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_tickers_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ticker_col = next((c for c in (reader.fieldnames or []) if c.strip().lower() == "ticker"), None)
            if not ticker_col:
                raise ValueError("CSV must have a 'ticker' column.")
            out = []
            for row in reader:
                t = (row.get(ticker_col) or "").strip()
                if t:
                    out.append(t)
            return out
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def clean_tickers(tickers: list[str]) -> list[str]:
    seen = set()
    out = []
    for t in tickers:
        tt = t.strip().upper()
        if tt and tt not in seen:
            out.append(tt)
            seen.add(tt)
    return out


def run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate PEAD figures (31–38) for many tickers. No viz scripts > 40.")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--tickers-file", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--panel-name", default="event_panel")
    ap.add_argument("--out-subdir", default="viz")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--continue-on-error", action="store_true")

    ap.add_argument("--skip-paths", action="store_true")              # 35
    ap.add_argument("--skip-event-cards", action="store_true")        # 36
    ap.add_argument("--skip-event-matrix", action="store_true")       # 37
    ap.add_argument("--skip-top-feature-timeline", action="store_true")  # 38
    args = ap.parse_args()

    tickers = []
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

    root = repo_root()
    scripts_dir = root / "scripts"
    py = sys.executable

    # 31–38 only
    ordered = [
        ("31_viz_event_overview.py", True),
        ("32_viz_surprise_vs_drift.py", True),
        ("33_viz_surprise_heatmaps.py", True),
        ("34_viz_feature_correlation.py", True),
        ("35_viz_event_paths.py", not args.skip_paths),
        ("37_viz_event_matrix.py", not args.skip_event_matrix),
        ("36_viz_event_cards.py", not args.skip_event_cards),
        ("38_viz_top_feature_timelines.py", not args.skip_top_feature_timeline),
    ]

    common = ["--panel-name", args.panel_name, "--out-subdir", args.out_subdir, "--dpi", str(args.dpi)]
    if args.data_dir:
        common += ["--data-dir", args.data_dir]

    for t in tickers:
        try:
            for fname, enabled in ordered:
                if not enabled:
                    continue
                sp = scripts_dir / fname
                if not sp.exists():
                    print(f"[WARN] {t}: missing {sp.name} — skipping.", flush=True)
                    continue
                run([py, "-u", str(sp), "--ticker", t] + common, cwd=root)

            print(f"[OK] {t}: figures complete", flush=True)
        except Exception as e:
            print(f"[ERROR] {t}: {e}", flush=True)
            if not args.continue_on_error:
                raise

    print("[OK] All requested figures complete.", flush=True)


if __name__ == "__main__":
    main()
