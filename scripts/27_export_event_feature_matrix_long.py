#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
27_export_event_feature_matrix_long.py

Build a long-format event-level dataset across tickers and export correlations.

Per ticker sources:
  - data/{T}/events/event_panel_numeric.csv  (targets + surprises + returns)
  - data/{T}/financials/key_metrics_quarter.csv
  - data/{T}/financials/ratios_quarter.csv

Logic:
  - Filter event_panel by earnings_date in [start, end]
  - Keep the FIRST N events (chronological) that have BOTH key_metrics + ratios rows
    (N default=16 via --n-events)
  - For each kept earnings event, create 4 cases:
      react   : y = retpct_react_vs_pre
      drift5  : y = retpct_drift_5_vs_react
      drift10 : y = retpct_drift_10_vs_react
      drift20 : y = retpct_drift_20_vs_react

Features:
  - ALL columns from key_metrics_quarter.csv (prefixed km_)
  - ALL columns from ratios_quarter.csv      (prefixed rt_)
  - Plus surprises from event_panel if present:
      eps_surprise_pct, revenue_surprise_pct

Outputs (defaults under data/_derived):
  - event_feature_matrix_long_train.csv
  - event_feature_matrix_long_train_skips.csv
  - event_feature_corrs_train.csv                 (pooled Pearson corr(feature,y) by case)
  - event_feature_corrs_by_ticker_train.csv       (per-ticker corr(feature,y) by case)
  - event_feature_corrs_companyavg_train.csv      (mean abs(per-ticker corr) by case)

Printing:
  - ONE combined “hits” printout:
      HIT if pooled_abs >= --pooled-thresh (default 0.10)
      RESCUED if pooled_abs < pooled-thresh AND mean_abs >= --rescue-thresh (default 0.30)
    Print line includes pooled_corr/pooled_abs AND mean_abs for the same case.

Notes / risk controls:
  - Drops year-like fields (calendarYear/fiscalYear)
  - Drops identifier-ish / time-ish fields (period, transcript_dir, fiscal_period_end)
  - Uses a safe Pearson implementation that returns NaN if std ~ 0, avoiding numpy RuntimeWarnings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SURPRISE_COLS = ["eps_surprise_pct", "revenue_surprise_pct"]

TARGET_COLS = {
    "react": "retpct_react_vs_pre",
    "drift5": "retpct_drift_5_vs_react",
    "drift10": "retpct_drift_10_vs_react",
    "drift20": "retpct_drift_20_vs_react",
}
CASES_ORDER = ["react", "drift5", "drift10", "drift20"]


# ---- Drop “year stuff” + other identifier/time-ish fields (leakage-ish / non-features) ----
DROP_FEATURES = {
    # year proxies
    "km_fiscalYear",
    "km_calendarYear",
    "rt_fiscalYear",
    "rt_calendarYear",
    # identifier-ish / non-numeric-ish fields (often strings)
    "km_period",
    "rt_period",
    "km_transcript_dir",
    "rt_transcript_dir",
    "km_fiscal_period_end",
    "rt_fiscal_period_end",
    # just in case older versions had these
    "x_react",
}


def infer_tickers(data_dir: Path) -> list[str]:
    out: list[str] = []
    for p in data_dir.iterdir():
        if p.is_dir() and p.name.isupper() and 1 <= len(p.name) <= 6 and not p.name.startswith("_"):
            out.append(p.name)
    return sorted(out)


def _to_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _prefix_cols(df: pd.DataFrame, prefix: str, keep: set[str]) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        if c in keep:
            continue
        ren[c] = f"{prefix}{c}"
    return df.rename(columns=ren)


def _as_num(s: pd.Series) -> pd.Series:
    # converts pd.NA / junk to np.nan float
    return pd.to_numeric(s, errors="coerce")


def _select_feature_cols(long_df: pd.DataFrame, key_cols: set[str]) -> list[str]:
    # candidate features = everything except keys and explicit drops
    return [c for c in long_df.columns if c not in key_cols and c not in DROP_FEATURES]


def _pearson_corr_pairwise(x: pd.Series, y: pd.Series) -> float:
    """
    Pearson corr using pairwise complete obs.
    Returns np.nan if <3 pairs OR if either series is (near) constant (std ~ 0).
    Avoids numpy RuntimeWarning: invalid value encountered in divide.
    """
    x = _as_num(x)
    y = _as_num(y)
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return np.nan

    xx = x[m].to_numpy(dtype=float)
    yy = y[m].to_numpy(dtype=float)

    sx = np.std(xx)
    sy = np.std(yy)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx < 1e-12 or sy < 1e-12:
        return np.nan

    return float(np.corrcoef(xx, yy)[0, 1])


def _compute_corr_pooled(long_df: pd.DataFrame, round_decimals: int = 3) -> pd.DataFrame:
    """
    Wide pooled correlation table (no n columns):
      feature,
      react_corr, react_abs,
      drift5_corr, drift5_abs,
      drift10_corr, drift10_abs,
      drift20_corr, drift20_abs
    """
    req = {"ticker", "earnings_date", "case", "y"}
    missing = req - set(long_df.columns)
    if missing:
        raise ValueError(f"long_df missing required cols: {sorted(missing)}")

    key_cols = {"ticker", "earnings_date", "case", "y"}
    feature_cols = _select_feature_cols(long_df, key_cols)

    rows = []
    for case in CASES_ORDER:
        sub = long_df[long_df["case"] == case].copy()
        for feat in feature_cols:
            corr = _pearson_corr_pairwise(sub[feat], sub["y"])
            rows.append(
                {
                    "feature": feat,
                    "case": case,
                    "corr": corr,
                    "abs_corr": abs(corr) if not np.isnan(corr) else np.nan,
                }
            )

    corr_long = pd.DataFrame(rows)
    wide_corr = corr_long.pivot(index="feature", columns="case", values="corr")
    wide_abs = corr_long.pivot(index="feature", columns="case", values="abs_corr")

    out = pd.DataFrame(index=sorted(wide_corr.index))
    for case in CASES_ORDER:
        out[f"{case}_corr"] = wide_corr.get(case)
        out[f"{case}_abs"] = wide_abs.get(case)

    out = out.reset_index().rename(columns={"index": "feature"})

    for c in out.columns:
        if c.endswith("_corr") or c.endswith("_abs"):
            out[c] = pd.to_numeric(out[c], errors="coerce").round(round_decimals)

    return out


def _compute_corr_by_ticker(long_df: pd.DataFrame, round_decimals: int = 3) -> pd.DataFrame:
    """
    Long per-ticker correlation table (no n columns):
      ticker, case, feature, corr, abs_corr
    """
    key_cols = {"ticker", "earnings_date", "case", "y"}
    feature_cols = _select_feature_cols(long_df, key_cols)

    rows = []
    for case in CASES_ORDER:
        sub_case = long_df[long_df["case"] == case].copy()
        for tkr, g in sub_case.groupby("ticker", sort=True):
            for feat in feature_cols:
                corr = _pearson_corr_pairwise(g[feat], g["y"])
                rows.append(
                    {
                        "ticker": tkr,
                        "case": case,
                        "feature": feat,
                        "corr": corr,
                        "abs_corr": abs(corr) if not np.isnan(corr) else np.nan,
                    }
                )

    out = pd.DataFrame(rows)
    out["corr"] = pd.to_numeric(out["corr"], errors="coerce").round(round_decimals)
    out["abs_corr"] = pd.to_numeric(out["abs_corr"], errors="coerce").round(round_decimals)
    return out


def _compute_company_mean_abs(by_ticker: pd.DataFrame, round_decimals: int = 3) -> pd.DataFrame:
    """
    Wide company-mean(abs) table (mean across tickers, per case):
      feature,
      react_mean_abs, drift5_mean_abs, drift10_mean_abs, drift20_mean_abs
    """
    frames = []
    for case in CASES_ORDER:
        sub = by_ticker[by_ticker["case"] == case].copy()
        grp = sub.groupby("feature", sort=True)["abs_corr"].mean()
        tmp = grp.reset_index().rename(columns={"abs_corr": f"{case}_mean_abs"})
        frames.append(tmp)

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="feature", how="outer")

    for c in out.columns:
        if c.endswith("_mean_abs"):
            out[c] = pd.to_numeric(out[c], errors="coerce").round(round_decimals)

    return out.sort_values("feature").reset_index(drop=True)


def _rank_in_case_pooled(corr_df: pd.DataFrame, feature: str, case: str) -> tuple[int | None, int]:
    abs_col = f"{case}_abs"
    tmp = corr_df[["feature", abs_col]].copy()
    tmp[abs_col] = pd.to_numeric(tmp[abs_col], errors="coerce")
    tmp = tmp.dropna(subset=[abs_col]).sort_values(abs_col, ascending=False).reset_index(drop=True)
    total = int(len(tmp))
    hit = tmp.index[tmp["feature"] == feature]
    if len(hit) == 0:
        return None, total
    return int(hit[0] + 1), total


def _print_surprise_diagnostics(pooled: pd.DataFrame) -> None:
    print("\n[SURPRISE DIAGNOSTICS] (pooled) ranking by abs(corr):")
    for feat in DEFAULT_SURPRISE_COLS:
        if feat not in set(pooled["feature"].tolist()):
            print(f"\n  {feat}: not present")
            continue

        r = pooled[pooled["feature"] == feat].iloc[0]
        print(f"\n  {feat}:")
        for case in CASES_ORDER:
            corr_col = f"{case}_corr"
            abs_col = f"{case}_abs"
            rank, total = _rank_in_case_pooled(pooled, feat, case)
            rank_s = f"#{rank}/{total}" if rank is not None else f"(no rank; total={total})"
            print(f"    - {case:6s}: corr={r.get(corr_col)} abs={r.get(abs_col)} rank={rank_s}")


def _print_hits_one_table(
    summary_df: pd.DataFrame,
    pooled_thresh: float = 0.10,
    rescue_thresh: float = 0.30,
) -> None:
    """
    ONE combined printout:
      HIT if pooled_abs >= pooled_thresh
      RESCUED if pooled_abs < pooled_thresh AND mean_abs >= rescue_thresh
    Includes mean_abs alongside pooled stats.
    """
    print(
        f"\n[HITS] pooled_abs >= {pooled_thresh:.3f}  OR  (pooled_abs < {pooled_thresh:.3f} AND mean_abs >= {rescue_thresh:.3f})"
    )

    for case in CASES_ORDER:
        corr_col = f"{case}_corr"
        abs_col = f"{case}_abs"
        mean_col = f"{case}_mean_abs"

        # if mean_col missing for some reason, still print pooled hits
        cols = ["feature", corr_col, abs_col] + ([mean_col] if mean_col in summary_df.columns else [])
        tmp = summary_df[cols].copy()
        tmp[abs_col] = pd.to_numeric(tmp[abs_col], errors="coerce")
        pooled_hit = tmp[abs_col] >= pooled_thresh

        if mean_col in tmp.columns:
            tmp[mean_col] = pd.to_numeric(tmp[mean_col], errors="coerce")
            rescued = (tmp[abs_col] < pooled_thresh) & (tmp[mean_col] >= rescue_thresh)
        else:
            rescued = pd.Series([False] * len(tmp))

        keep = pooled_hit | rescued
        tmp = tmp[keep].copy()
        tmp["tag"] = np.where(pooled_hit[keep].to_numpy(), "HIT", "RESCUED")

        sort_cols = [abs_col] + ([mean_col] if mean_col in tmp.columns else [])
        tmp = tmp.sort_values(sort_cols, ascending=False)

        print(f"\n  {case}: {len(tmp)} features")
        for _, r in tmp.iterrows():
            if mean_col in tmp.columns:
                print(
                    f"    [{r['tag']}] {r['feature']}: pooled_corr={r[corr_col]} pooled_abs={r[abs_col]} mean_abs={r[mean_col]}"
                )
            else:
                print(f"    [{r['tag']}] {r['feature']}: pooled_corr={r[corr_col]} pooled_abs={r[abs_col]}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tickers", nargs="*", default=None, help="If omitted, infer from data-dir subfolders")
    ap.add_argument("--panel-name", default="event_panel_numeric", help="events/{panel-name}.csv")

    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2024-12-31")

    ap.add_argument(
        "--n-events",
        type=int,
        default=16,
        help="Keep the FIRST N earnings events per ticker (chronological) that have financial rows.",
    )

    # outputs under data/_derived by default
    ap.add_argument("--out", default="data/_derived/event_feature_matrix_long_train.csv")
    ap.add_argument("--skip-log", default="data/_derived/event_feature_matrix_long_train_skips.csv")

    ap.add_argument("--out-corr", default="data/_derived/event_feature_corrs_train.csv")
    ap.add_argument("--out-corr-by-ticker", default="data/_derived/event_feature_corrs_by_ticker_train.csv")
    ap.add_argument("--out-corr-companyavg", default="data/_derived/event_feature_corrs_companyavg_train.csv")

    ap.add_argument("--round", type=int, default=3, help="Round correlation outputs to N decimals (default 3).")

    ap.add_argument(
        "--require-all-targets",
        action="store_true",
        help="If set, skip an event unless ALL 4 target cols are present and non-null in event_panel.",
    )

    ap.add_argument("--pooled-thresh", type=float, default=0.10, help="HIT if pooled_abs >= this threshold.")
    ap.add_argument("--rescue-thresh", type=float, default=0.30, help="RESCUED if mean_abs >= this AND pooled_abs < pooled_thresh.")

    ap.add_argument("--print-surprise-diagnostics", action="store_true", help="Print pooled ranking diagnostics for surprises.")

    args = ap.parse_args()

    root = Path(".").resolve()
    data_dir = (root / args.data_dir).resolve()

    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skip_path = (root / args.skip_log).resolve()
    skip_path.parent.mkdir(parents=True, exist_ok=True)

    corr_path = (root / args.out_corr).resolve()
    corr_path.parent.mkdir(parents=True, exist_ok=True)

    corr_by_ticker_path = (root / args.out_corr_by_ticker).resolve()
    corr_by_ticker_path.parent.mkdir(parents=True, exist_ok=True)

    corr_companyavg_path = (root / args.out_corr_companyavg).resolve()
    corr_companyavg_path.parent.mkdir(parents=True, exist_ok=True)

    tickers = args.tickers if args.tickers else infer_tickers(data_dir)

    start = pd.to_datetime(args.start).date()
    end = pd.to_datetime(args.end).date()

    all_rows: list[dict] = []
    skip_rows: list[dict] = []

    for tkr in tickers:
        panel_path = data_dir / tkr / "events" / f"{args.panel_name}.csv"
        km_path = data_dir / tkr / "financials" / "key_metrics_quarter.csv"
        rt_path = data_dir / tkr / "financials" / "ratios_quarter.csv"

        if not panel_path.exists():
            print(f"[WARN] {tkr}: missing {panel_path}")
            continue
        if not km_path.exists():
            print(f"[WARN] {tkr}: missing {km_path}")
            continue
        if not rt_path.exists():
            print(f"[WARN] {tkr}: missing {rt_path}")
            continue

        panel = pd.read_csv(panel_path)
        if "earnings_date" not in panel.columns:
            print(f"[WARN] {tkr}: event panel missing earnings_date ({panel_path.name})")
            continue

        missing_targets = [c for c in set(TARGET_COLS.values()) if c not in panel.columns]
        if missing_targets:
            print(f"[WARN] {tkr}: missing target cols in panel: {missing_targets}")
            continue

        panel["earnings_date"] = _to_date_series(panel["earnings_date"])
        panel = panel[(panel["earnings_date"] >= start) & (panel["earnings_date"] <= end)].copy()
        panel = panel.sort_values("earnings_date").reset_index(drop=True)

        if panel.empty:
            print(f"[WARN] {tkr}: no events in range {start}..{end}")
            continue

        if args.require_all_targets:
            need = list(TARGET_COLS.values())
            before = len(panel)
            panel = panel.dropna(subset=need).reset_index(drop=True)
            if panel.empty:
                print(f"[WARN] {tkr}: after require-all-targets, no usable events remain")
                continue
            if len(panel) != before:
                print(f"[INFO] {tkr}: require-all-targets dropped {before - len(panel)} events")

        km = pd.read_csv(km_path)
        rt = pd.read_csv(rt_path)

        if "symbol" in km.columns:
            km = km[km["symbol"].astype(str).str.upper() == tkr].copy()
        if "symbol" in rt.columns:
            rt = rt[rt["symbol"].astype(str).str.upper() == tkr].copy()

        if "earnings_date" not in km.columns or "earnings_date" not in rt.columns:
            print(f"[WARN] {tkr}: financials missing earnings_date key (km or rt). Cannot merge.")
            continue

        km["earnings_date"] = _to_date_series(km["earnings_date"])
        rt["earnings_date"] = _to_date_series(rt["earnings_date"])

        key_keep = {"ticker", "symbol", "earnings_date"}
        km = _prefix_cols(km, "km_", keep=key_keep)
        rt = _prefix_cols(rt, "rt_", keep=key_keep)

        km_dates = set(km["earnings_date"].dropna().tolist())
        rt_dates = set(rt["earnings_date"].dropna().tolist())

        kept_events = 0
        skipped_events = 0

        for _, r in panel.iterrows():
            if kept_events >= args.n_events:
                break

            d = r["earnings_date"]
            if pd.isna(d):
                continue

            has_km = d in km_dates
            has_rt = d in rt_dates

            if not has_km or not has_rt:
                skipped_events += 1
                msg = []
                if not has_km:
                    msg.append("missing key_metrics row")
                if not has_rt:
                    msg.append("missing ratios row")
                reason = "; ".join(msg)
                print(f"[ERROR] {tkr}: {d} -> {reason}  | SKIPPING EVENT")
                skip_rows.append({"ticker": tkr, "earnings_date": d, "reason": reason})
                continue

            # grab first row if duplicates exist (consistent with your prior versions)
            km_row = km[km["earnings_date"] == d].iloc[0].to_dict()
            rt_row = rt[rt["earnings_date"] == d].iloc[0].to_dict()

            for case, ycol in TARGET_COLS.items():
                y = r.get(ycol)

                row = {
                    "ticker": tkr,
                    "earnings_date": d,
                    "case": case,
                    "y": y,
                }

                # surprises from panel
                for c in DEFAULT_SURPRISE_COLS:
                    if c in panel.columns:
                        row[c] = r.get(c)

                # include ALL financial columns, but drop identifiers + year stuff
                for k, v in km_row.items():
                    if k in ("symbol", "ticker", "earnings_date"):
                        continue
                    if k in DROP_FEATURES:
                        continue
                    row[k] = v

                for k, v in rt_row.items():
                    if k in ("symbol", "ticker", "earnings_date"):
                        continue
                    if k in DROP_FEATURES:
                        continue
                    row[k] = v

                all_rows.append(row)

            kept_events += 1

        if kept_events < args.n_events:
            print(f"[WARN] {tkr}: kept only {kept_events}/{args.n_events} events (skipped {skipped_events})")

        print(f"[OK] {tkr}: kept_events={kept_events} -> rows added={kept_events * 4}")

    if not all_rows:
        raise SystemExit("No rows produced. Check data-dir/tickers/panel-name/date range and financial availability.")

    long_df = pd.DataFrame(all_rows)
    long_df = long_df.dropna(subset=["y"]).reset_index(drop=True)

    long_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote long dataset -> {out_path} (rows={len(long_df)}, cols={len(long_df.columns)})")

    if skip_rows:
        pd.DataFrame(skip_rows).to_csv(skip_path, index=False)
        print(f"[WARN] Wrote skip log -> {skip_path} (rows={len(skip_rows)})")
    else:
        pd.DataFrame(columns=["ticker", "earnings_date", "reason"]).to_csv(skip_path, index=False)
        print(f"[OK] No skips. Wrote empty skip log -> {skip_path}")

    # --- correlations (no warnings, no n in outputs) ---
    pooled = _compute_corr_pooled(long_df, round_decimals=args.round)
    pooled.to_csv(corr_path, index=False)
    print(f"[OK] Wrote pooled correlations -> {corr_path} (rows={len(pooled)}, cols={len(pooled.columns)})")

    by_ticker = _compute_corr_by_ticker(long_df, round_decimals=args.round)
    by_ticker.to_csv(corr_by_ticker_path, index=False)
    print(f"[OK] Wrote per-ticker correlations -> {corr_by_ticker_path} (rows={len(by_ticker)}, cols={len(by_ticker.columns)})")

    company_mean = _compute_company_mean_abs(by_ticker, round_decimals=args.round)
    company_mean.to_csv(corr_companyavg_path, index=False)
    print(f"[OK] Wrote company-mean(abs) correlations -> {corr_companyavg_path} (rows={len(company_mean)}, cols={len(company_mean.columns)})")

    # optionally print surprise diagnostics
    if args.print_surprise_diagnostics:
        _print_surprise_diagnostics(pooled)

    # ONE combined printout: pooled hits, plus rescued-by-mean-abs
    summary = pooled.merge(company_mean, on="feature", how="left")
    _print_hits_one_table(summary, pooled_thresh=float(args.pooled_thresh), rescue_thresh=float(args.rescue_thresh))


if __name__ == "__main__":
    main()
