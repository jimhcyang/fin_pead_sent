#!/usr/bin/env python3
# scripts/23_compute_event_returns.py
#
# Event-study style outputs:
#   - adjusted close levels
#   - price deltas in dollars
#   - simple returns in percent (%)
#
# This aligns with standard event-study / PEAD practice where drift is measured
# via (cumulative) returns and abnormal returns over post-event windows. :contentReference[oaicite:1]{index=1}

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    return repo_root() / "data"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_price_path(data_dir: Path, ticker: str, name: str) -> pd.DataFrame:
    p = data_dir / ticker / "events" / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing price path: {p} (run 22_extract_event_price_path.py)")
    return pd.read_csv(p)


def _ret_pct(a: float, b: float) -> float:
    """Simple return in percent: 100*(b/a - 1)."""
    if not (np.isfinite(a) and np.isfinite(b)) or a == 0:
        return float("nan")
    return 100.0 * (b / a - 1.0)


def _delta(a: float, b: float) -> float:
    """Dollar difference: b - a."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return b - a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--price-path-name", default="event_price_path")
    ap.add_argument("--out-name", default="event_returns")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    df = load_price_path(data_dir, ticker, args.price_path_name)

    rows = []
    for _, r in df.iterrows():
        pre = float(pd.to_numeric(r.get("pre_adj_close"), errors="coerce"))
        react = float(pd.to_numeric(r.get("react_adj_close"), errors="coerce"))
        p5 = float(pd.to_numeric(r.get("adj_close_p5"), errors="coerce"))
        p10 = float(pd.to_numeric(r.get("adj_close_p10"), errors="coerce"))
        p20 = float(pd.to_numeric(r.get("adj_close_p20"), errors="coerce"))

        rows.append(
            {
                "ticker": r.get("ticker", ticker),
                "earnings_date": r["earnings_date"],
                "announce_timing": r["announce_timing"],
                "pre_date": r["pre_date"],
                "react_date": r["react_date"],
                "d5": r["d5"],
                "d10": r["d10"],
                "d20": r["d20"],

                # price levels
                "pre_adj_close": pre,
                "react_adj_close": react,
                "adj_close_p5": p5,
                "adj_close_p10": p10,
                "adj_close_p20": p20,

                # deltas vs pre (levels)
                "delta_react_vs_pre": _delta(pre, react),
                "delta_p5_vs_pre": _delta(pre, p5),
                "delta_p10_vs_pre": _delta(pre, p10),
                "delta_p20_vs_pre": _delta(pre, p20),

                # drift deltas vs react (post-event drift channel)
                "delta_drift_5_vs_react": _delta(react, p5),
                "delta_drift_10_vs_react": _delta(react, p10),
                "delta_drift_20_vs_react": _delta(react, p20),

                # returns (%) vs pre
                "retpct_react_vs_pre": _ret_pct(pre, react),
                "retpct_p5_vs_pre": _ret_pct(pre, p5),
                "retpct_p10_vs_pre": _ret_pct(pre, p10),
                "retpct_p20_vs_pre": _ret_pct(pre, p20),

                # drift returns (%) vs react
                "retpct_drift_5_vs_react": _ret_pct(react, p5),
                "retpct_drift_10_vs_react": _ret_pct(react, p10),
                "retpct_drift_20_vs_react": _ret_pct(react, p20),
            }
        )

    out = pd.DataFrame(rows)

    out_dir = data_dir / ticker / "events"
    ensure_dir(out_dir)
    out_csv = out_dir / f"{args.out_name}.csv"
    out.to_csv(out_csv, index=False)

    meta = {
        "ticker": ticker,
        "rows": int(out.shape[0]),
        "created_at_local": datetime.now().isoformat(),
        "notes": [
            "Minimal event-study outputs: price levels, $ deltas, and simple returns (%) only.",
            "Drift returns (post-event) are measured vs react_date close (retpct_drift_*).",
        ],
    }
    (out_dir / f"{args.out_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] {ticker}: wrote {out_csv} ({out.shape[0]} rows)")


if __name__ == "__main__":
    main()