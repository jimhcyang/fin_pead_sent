#!/usr/bin/env python3
# scripts/20_run_event_pipeline_all.py

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List

from _common import DEFAULT_20


def run(cmd: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--skip-market", action="store_true")
    ap.add_argument("--force-market-download", action="store_true", help="Re-download market even if file exists.")
    ap.add_argument("--market-rel", default="_tmp_market/spx/prices/yf_ohlcv_daily.csv")
    ap.add_argument("--market-end", default="2026-01-31", help="End date for market download (can exceed event end).")
    ap.add_argument("--pre-bdays", type=int, default=5)
    ap.add_argument("--post-bdays", type=int, default=10)
    ap.add_argument("--stable-period", choices=["quarter", "annual"], default="quarter")
    args = ap.parse_args()

    root = repo_root()
    scripts = root / "scripts"

    tickers = [t.upper() for t in (args.tickers or DEFAULT_20)]

    # ensure FMP key exists for stable features
    if not os.getenv("FMP_API_KEY"):
        print("[WARN] FMP_API_KEY not set. If stable files already exist on disk, it's fine. If not, run script 10 first.")

    # ensure market file exists (reuse your existing downloader)
    if not args.skip_market:
        market_path = args.data_dir / args.market_rel
        if market_path.exists() and not args.force_market_download:
            print(f"[INFO] Reusing existing market file: {market_path}")
        else:
            run(
                [
                    "python",
                    str(scripts / "25_yf_download_spx.py"),
                    "--data-dir",
                    str(args.data_dir),
                    "--symbol",
                    "^GSPC",
                    "--start",
                    args.start,
                    "--end",
                    args.market_end,
                    "--out-rel",
                    args.market_rel,
                ],
                cwd=root,
            )

    for tkr in tickers:
        run(["python", str(scripts / "11_build_event_windows.py"), "--ticker", tkr, "--data-dir", str(args.data_dir),
             "--pre-bdays", str(int(args.pre_bdays)), "--post-bdays", str(int(args.post_bdays))], cwd=root)

        run(["python", str(scripts / "12_extract_event_price_path.py"), "--ticker", tkr, "--data-dir", str(args.data_dir),
             "--pre-bdays", str(int(args.pre_bdays)), "--post-bdays", str(int(args.post_bdays))], cwd=root)

        run(["python", str(scripts / "13_compute_event_returns.py"), "--ticker", tkr, "--data-dir", str(args.data_dir)], cwd=root)

        run(["python", str(scripts / "14_fit_event_market_model.py"), "--ticker", tkr, "--data-dir", str(args.data_dir),
             "--market-rel", args.market_rel], cwd=root)

        run(["python", str(scripts / "15_compute_event_abnormal_returns.py"), "--ticker", tkr, "--data-dir", str(args.data_dir),
             "--market-rel", args.market_rel], cwd=root)

        run(["python", str(scripts / "16_merge_event_features_stable.py"), "--ticker", tkr, "--data-dir", str(args.data_dir),
             "--period", args.stable_period], cwd=root)

        run(["python", str(scripts / "17_build_event_panel.py"), "--ticker", tkr, "--data-dir", str(args.data_dir)], cwd=root)

    # once: long export + corrs
    run(["python", str(scripts / "18_export_event_long.py"), "--data-dir", str(args.data_dir)], cwd=root)
    run(["python", str(scripts / "19_event_feature_corrs.py")], cwd=root)

    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
