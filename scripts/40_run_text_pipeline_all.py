#!/usr/bin/env python3
# scripts/40_run_text_pipeline_all.py
"""
Convenience wrapper:
1) (optional) extract text units (news + transcripts) for tickers
2) (optional) score units with LM dictionary
3) merge per-event LM features into a single wide table

Outputs:
  data/_derived/text/event_text_features.csv
  data/_derived/text/event_text_features.meta.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from _common import DEFAULT_20


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_tickers_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
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


def run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        raise RuntimeError(f"Command failed (exit={r.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run text extraction + LM scoring + merge features.")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--tickers-file", default=None)
    ap.add_argument("--max-tickers", type=int, default=None)

    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--min-text-len", type=int, default=20)
    ap.add_argument("--min-tokens", type=int, default=20)

    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-operator", action="store_true")

    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    root = repo_root()

    # tickers
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
        tickers = tickers[: int(args.max_tickers)]

    py = sys.executable

    # step 1: extract units
    if not args.skip_extract:
        cmd = [
            py, "-u", str(root / "scripts" / "31_extract_text_units.py"),
            "--data-dir", str(data_dir),
            "--min-text-len", str(int(args.min_text_len)),
            "--tickers",
            *tickers,
        ]
        run(cmd, cwd=root)

    # step 2: score LM
    if not args.skip_score:
        cmd_tr = [
            py, "-u", str(root / "scripts" / "32_score_text_LM_transcripts.py"),
            "--data-dir", str(data_dir),
            "--min-tokens", str(int(args.min_tokens)),
            "--tickers",
            *tickers,
        ]
        if args.overwrite:
            cmd_tr.append("--overwrite")
        if args.keep_operator:
            cmd_tr.append("--keep-operator")
        run(cmd_tr, cwd=root)

        cmd_nw = [
            py, "-u", str(root / "scripts" / "33_score_text_LM_news.py"),
            "--data-dir", str(data_dir),
            "--min-tokens", str(int(args.min_tokens)),
            "--tickers",
            *tickers,
        ]
        if args.overwrite:
            cmd_nw.append("--overwrite")
        run(cmd_nw, cwd=root)

    # step 3: merge features
    merged_parts: list[pd.DataFrame] = []
    per_ticker_meta = {}

    for t in tickers:
        t = t.upper()
        ev_path = data_dir / t / "events" / "event_windows.csv"
        if not ev_path.exists():
            print(f"[WARN] {t}: missing {ev_path} (run the event pipeline first)")
            continue
        ev = pd.read_csv(ev_path)

        keys = ["ticker", "earnings_date", "day0_date"]
        for k in keys:
            if k not in ev.columns:
                raise ValueError(f"{ev_path} missing column: {k}")

        # optional feature tables
        tr_path = data_dir / t / "events" / "text_lm_transcripts_event_wide.csv"
        nw_path = data_dir / t / "events" / "text_lm_news_event_wide.csv"

        tr = pd.read_csv(tr_path) if tr_path.exists() else pd.DataFrame(columns=keys)
        nw = pd.read_csv(nw_path) if nw_path.exists() else pd.DataFrame(columns=keys)

        out = ev.merge(tr, on=keys, how="left").merge(nw, on=keys, how="left")
        merged_parts.append(out)

        per_ticker_meta[t] = {
            "events": int(ev.shape[0]),
            "has_tr": bool(tr_path.exists()),
            "has_nw": bool(nw_path.exists()),
        }

    out_all = pd.concat(merged_parts, ignore_index=True) if merged_parts else pd.DataFrame()
    out_dir = data_dir / "_derived" / "text"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "event_text_features.csv"
    out_all.to_csv(out_path, index=False)

    meta = {
        "tickers": tickers,
        "rows": int(out_all.shape[0]),
        "cols": int(out_all.shape[1]) if not out_all.empty else 0,
        "output": str(out_path),
        "per_ticker": per_ticker_meta,
        "min_text_len": int(args.min_text_len),
        "min_tokens": int(args.min_tokens),
        "overwrite": bool(args.overwrite),
        "keep_operator": bool(args.keep_operator),
        "skip_extract": bool(args.skip_extract),
        "skip_score": bool(args.skip_score),
    }
    (out_dir / "event_text_features.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote merged event text features -> {out_path} rows={meta['rows']:,} cols={meta['cols']:,}")


if __name__ == "__main__":
    main()
