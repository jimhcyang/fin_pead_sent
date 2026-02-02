#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/41_fit_event_ml_baselines.py (v2)

Target:
  drift_car_mm_pct

Leakage prevention:
  drops any feature starting with "drift_" (future drift window).

Split:
  per ticker: last N valid-target events = test

Outputs:
  data/_derived/event_ml_baselines/metrics_*.csv
  data/_derived/event_ml_baselines/preds_*.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_panel(data_dir: Path, ticker: str, name: str = "event_ml_panel") -> pd.DataFrame:
    p = data_dir / ticker / "events" / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p} (run scripts/24_merge_event_fundamentals.py first)")
    df = pd.read_csv(p)
    df["ticker"] = ticker.upper()
    return df


def infer_tickers(data_dir: Path) -> List[str]:
    out = []
    for p in sorted(data_dir.glob("*")):
        if p.is_dir() and (p / "events" / "event_ml_panel.csv").exists():
            out.append(p.name.upper())
    return out


def select_xy(df: pd.DataFrame, ycol: str, include_time: bool) -> Tuple[pd.DataFrame, pd.Series]:
    if ycol not in df.columns:
        raise ValueError(f"Target '{ycol}' not found.")

    y = pd.to_numeric(df[ycol], errors="coerce")

    drop_cols = {ycol}

    # leakage: anything computed over FUTURE drift window
    for c in df.columns:
        if c.startswith("drift_"):
            drop_cols.add(c)

    # future metadata
    if "next_event_date" in df.columns:
        drop_cols.add("next_event_date")

    # drop raw date-ish strings
    date_like = [c for c in df.columns if c.endswith("_date") or c in ("event_date",)]
    drop_cols.update(date_like)

    df2 = df.copy()
    if include_time and "event_date" in df2.columns:
        dt = pd.to_datetime(df2["event_date"], errors="coerce")
        df2["time_ordinal"] = dt.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

    X = df2.drop(columns=[c for c in drop_cols if c in df2.columns], errors="ignore")

    # drop any remaining object columns except ticker
    for c in list(X.columns):
        if c != "ticker" and X[c].dtype == "object":
            X = X.drop(columns=[c])

    if "ticker" not in X.columns and "ticker" in df2.columns:
        X["ticker"] = df2["ticker"]

    return X, y


def make_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]  # usually just ticker
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols


def split_by_ticker(
    df: pd.DataFrame,
    ycol: str,
    test_n: int,
    skip_last_k: int,
    include_time: bool,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []

    for tkr, g in df.groupby("ticker", sort=True):
        sort_key = "event_idx" if "event_idx" in g.columns else "event_date"
        g = g.sort_values(sort_key).reset_index(drop=True)

        y_all = pd.to_numeric(g[ycol], errors="coerce")
        g = g.loc[y_all.notna()].copy()
        if g.empty:
            continue

        if skip_last_k > 0:
            g = g.iloc[:-skip_last_k].copy()
            if g.empty:
                continue

        if len(g) <= test_n:
            X, y = select_xy(g, ycol=ycol, include_time=include_time)
            X_tr_list.append(X)
            y_tr_list.append(y)
            continue

        g_train = g.iloc[:-test_n].copy()
        g_test = g.iloc[-test_n:].copy()

        Xtr, ytr = select_xy(g_train, ycol=ycol, include_time=include_time)
        Xte, yte = select_xy(g_test, ycol=ycol, include_time=include_time)

        X_tr_list.append(Xtr); y_tr_list.append(ytr)
        X_te_list.append(Xte); y_te_list.append(yte)

    if not X_tr_list:
        raise RuntimeError("No training data after filtering.")

    X_train = pd.concat(X_tr_list, axis=0, ignore_index=True)
    y_train = pd.concat(y_tr_list, axis=0, ignore_index=True)

    X_test = pd.concat(X_te_list, axis=0, ignore_index=True) if X_te_list else pd.DataFrame()
    y_test = pd.concat(y_te_list, axis=0, ignore_index=True) if y_te_list else pd.Series(dtype=float)

    return X_train, y_train, X_test, y_test


def eval_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)   # older sklearn doesn't support squared=
    rmse = float(np.sqrt(mse))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--ycol", default="drift_car_mm_pct")
    ap.add_argument("--test-n", type=int, default=4)
    ap.add_argument("--skip-last-k", type=int, default=0)
    ap.add_argument("--include-time", action="store_true")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    tickers = [t.upper() for t in (args.tickers or infer_tickers(args.data_dir))]
    if not tickers:
        raise RuntimeError("No tickers found with data/{T}/events/event_ml_panel.csv")

    df = pd.concat([load_panel(args.data_dir, t) for t in tickers], axis=0, ignore_index=True)

    X_train, y_train, X_test, y_test = split_by_ticker(
        df=df,
        ycol=args.ycol,
        test_n=args.test_n,
        skip_last_k=args.skip_last_k,
        include_time=args.include_time,
    )

    pre, _, _ = make_preprocessor(X_train)

    models = [
        ("martingale_zero", None),
        ("train_mean", None),
        ("ridge", Ridge(alpha=1.0, random_state=args.random_state)),
        ("hgb", HistGradientBoostingRegressor(random_state=args.random_state)),
        ("rf", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=args.random_state,
            n_jobs=1,
        )),
    ]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (args.data_dir / "_derived" / "event_ml_baselines")
    ensure_dir(out_dir)

    metrics_rows = []
    preds_rows = []

    y_true = y_test.to_numpy(dtype=float) if not X_test.empty else None

    for name, model in models:
        if X_test.empty:
            yhat = np.array([])
        elif name == "martingale_zero":
            yhat = np.zeros_like(y_true)
        elif name == "train_mean":
            mu = float(np.nanmean(y_train.to_numpy(dtype=float)))
            yhat = np.full_like(y_true, mu)
        else:
            pipe = Pipeline([("pre", pre), ("model", model)])
            pipe.fit(X_train, y_train.to_numpy(dtype=float))
            yhat = pipe.predict(X_test)

        row = {"model": name, "n_train": int(len(y_train)), "n_test": int(len(y_test))}
        if not X_test.empty:
            row.update(eval_preds(y_true, yhat))
            for i in range(len(y_true)):
                preds_rows.append({"model": name, "y_true": float(y_true[i]), "y_pred": float(yhat[i])})
        else:
            row.update({"mae": np.nan, "rmse": np.nan, "r2": np.nan})

        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows).sort_values(["rmse", "mae"], ascending=[True, True])
    metrics_path = out_dir / f"metrics_{stamp}.csv"
    metrics.to_csv(metrics_path, index=False)

    preds = pd.DataFrame(preds_rows)
    preds_path = out_dir / f"preds_{stamp}.csv"
    preds.to_csv(preds_path, index=False)

    print("\n=== Event-ML results ===")
    print(metrics.to_string(index=False))
    print(f"\n[OK] wrote: {metrics_path}")
    print(f"[OK] wrote: {preds_path}")


if __name__ == "__main__":
    main()