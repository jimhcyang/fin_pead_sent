#!/usr/bin/env python3
# scripts/26_compute_abnormal_returns.py
#
# Computes market returns (%) using SPX adj_close and adds abnormal returns (%)
# as stock - market for matched anchor definitions.
#
# Close-only event study / PEAD convention: returns are close-to-close and
# drift is measured post-event over subsequent days. :contentReference[oaicite:2]{index=2}

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


def load_panel_numeric(data_dir: Path, ticker: str, name: str) -> pd.DataFrame:
    p = data_dir / ticker / "events" / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing panel numeric: {p} (run 24_merge_event_fundamentals.py)")
    return pd.read_csv(p)


def load_spx(data_dir: Path, rel: str) -> pd.DataFrame:
    p = data_dir / rel
    if not p.exists():
        raise FileNotFoundError(f"Missing SPX file: {p}")
    df = pd.read_csv(p)

    required = {"date", "adj_close"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"SPX file missing columns {sorted(missing)}: {p}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _ret_pct(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or a == 0:
        return float("nan")
    return 100.0 * (b / a - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--panel-name", default="event_panel_numeric")
    ap.add_argument("--spx-rel", default="_tmp_market/spx/prices/yf_ohlcv_daily.csv")
    ap.add_argument("--out-name", default="event_panel")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    ev = load_panel_numeric(data_dir, ticker, args.panel_name)
    spx = load_spx(data_dir, args.spx_rel)

    spx_close = dict(zip(spx["date"], spx["adj_close"]))

    def mclose(d: str) -> float:
        return float(spx_close.get(d, np.nan))

    out_rows = []
    for _, r in ev.iterrows():
        pre_d = r.get("pre_date")
        react_d = r.get("react_date")
        d5 = r.get("d5")
        d10 = r.get("d10")
        d20 = r.get("d20")

        m_pre = mclose(pre_d)
        m_react = mclose(react_d)
        m_5 = mclose(d5)
        m_10 = mclose(d10)
        m_20 = mclose(d20)

        # market returns (%) matching the stock definitions in script 23
        mkt_retpct_react_vs_pre = _ret_pct(m_pre, m_react)
        mkt_retpct_p5_vs_pre = _ret_pct(m_pre, m_5)
        mkt_retpct_p10_vs_pre = _ret_pct(m_pre, m_10)
        mkt_retpct_p20_vs_pre = _ret_pct(m_pre, m_20)

        mkt_retpct_drift_5_vs_react = _ret_pct(m_react, m_5)
        mkt_retpct_drift_10_vs_react = _ret_pct(m_react, m_10)
        mkt_retpct_drift_20_vs_react = _ret_pct(m_react, m_20)

        def abn(stock_col: str, mkt_val: float) -> float:
            sv = float(pd.to_numeric(r.get(stock_col), errors="coerce"))
            if not (np.isfinite(sv) and np.isfinite(mkt_val)):
                return float("nan")
            return sv - mkt_val

        out = dict(r)
        out.update(
            {
                # market levels (debuggable)
                "mkt_pre_adj_close": m_pre,
                "mkt_react_adj_close": m_react,
                "mkt_adj_close_p5": m_5,
                "mkt_adj_close_p10": m_10,
                "mkt_adj_close_p20": m_20,

                # market returns (%)
                "mkt_retpct_react_vs_pre": mkt_retpct_react_vs_pre,
                "mkt_retpct_p5_vs_pre": mkt_retpct_p5_vs_pre,
                "mkt_retpct_p10_vs_pre": mkt_retpct_p10_vs_pre,
                "mkt_retpct_p20_vs_pre": mkt_retpct_p20_vs_pre,
                "mkt_retpct_drift_5_vs_react": mkt_retpct_drift_5_vs_react,
                "mkt_retpct_drift_10_vs_react": mkt_retpct_drift_10_vs_react,
                "mkt_retpct_drift_20_vs_react": mkt_retpct_drift_20_vs_react,

                # abnormal returns (%): stock - market
                "abn_retpct_react_vs_pre": abn("retpct_react_vs_pre", mkt_retpct_react_vs_pre),
                "abn_retpct_p5_vs_pre": abn("retpct_p5_vs_pre", mkt_retpct_p5_vs_pre),
                "abn_retpct_p10_vs_pre": abn("retpct_p10_vs_pre", mkt_retpct_p10_vs_pre),
                "abn_retpct_p20_vs_pre": abn("retpct_p20_vs_pre", mkt_retpct_p20_vs_pre),
                "abn_retpct_drift_5_vs_react": abn("retpct_drift_5_vs_react", mkt_retpct_drift_5_vs_react),
                "abn_retpct_drift_10_vs_react": abn("retpct_drift_10_vs_react", mkt_retpct_drift_10_vs_react),
                "abn_retpct_drift_20_vs_react": abn("retpct_drift_20_vs_react", mkt_retpct_drift_20_vs_react),
            }
        )

        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)

    out_dir = data_dir / ticker / "events"
    ensure_dir(out_dir)
    out_csv = out_dir / f"{args.out_name}.csv"
    out_df.to_csv(out_csv, index=False)

    meta = {
        "ticker": ticker,
        "rows": int(out_df.shape[0]),
        "created_at_local": datetime.now().isoformat(),
        "spx_source": args.spx_rel,
        "notes": [
            "Close-only abnormal returns: simple returns (%) computed from adj_close.",
            "Abnormal returns are computed as stock - market for the same anchor dates.",
        ],
    }
    (out_dir / f"{args.out_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] {ticker}: wrote {out_csv} ({out_df.shape[0]} rows)")


if __name__ == "__main__":
    main()