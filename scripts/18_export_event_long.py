#!/usr/bin/env python3
# scripts/18_export_event_long.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from _eventlib import WINDOWS, ensure_dir


def infer_tickers(data_dir: Path) -> List[str]:
    out = []
    for p in data_dir.iterdir():
        if p.is_dir() and p.name.isupper() and (p / "events" / "event_panel.csv").exists():
            out.append(p.name)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--panel-name", default="event_panel")
    ap.add_argument("--out", default="data/_derived/event_long.csv")
    args = ap.parse_args()

    data_dir = args.data_dir
    tickers = [t.upper() for t in (args.tickers or infer_tickers(data_dir))]

    if not tickers:
        raise RuntimeError("No tickers found with events/event_panel.csv")

    all_rows = []
    for tkr in tickers:
        p = data_dir / tkr / "events" / f"{args.panel_name}.csv"
        df = pd.read_csv(p)
        df["ticker"] = tkr

        # features = all columns except the y columns we generate below
        for win_name in WINDOWS.keys():
            y_ret = pd.to_numeric(df.get(f"pct_{win_name}", np.nan), errors="coerce")
            y_car_s = pd.to_numeric(df.get(f"car_simple_{win_name}_pct", np.nan), errors="coerce")
            y_car_y = pd.to_numeric(df.get(f"car_mm_1y_{win_name}_pct", np.nan), errors="coerce")
            y_car_q = pd.to_numeric(df.get(f"car_mm_1q_{win_name}_pct", np.nan), errors="coerce")

            base = df.copy()
            base["window"] = win_name

            # We'll stack 4 y-types to keep one unified long file
            for ytype, y in [
                ("ret_pct", y_ret),
                ("car_simple_pct", y_car_s),
                ("car_mm_1y_pct", y_car_y),
                ("car_mm_1q_pct", y_car_q),
            ]:
                tmp = base.copy()
                tmp["y_type"] = ytype
                tmp["y"] = y
                all_rows.append(tmp)

    out = pd.concat(all_rows, axis=0, ignore_index=True)

    # drop columns that are explicitly y-defining to reduce duplication noise
    drop_like = [c for c in out.columns if c.startswith("pct_") or c.startswith("car_")]
    out = out.drop(columns=drop_like, errors="ignore")

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)

    print(f"[OK] wrote long dataset -> {out_path} (rows={len(out)}, cols={len(out.columns)})")


if __name__ == "__main__":
    main()
