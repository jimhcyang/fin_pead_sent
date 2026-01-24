#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import warnings
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Warning suppression (requested)
# -----------------------------
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"divide by zero encountered in matmul",
)

try:
    from matplotlib import MatplotlibDeprecationWarning
    warnings.filterwarnings(
        "ignore",
        category=MatplotlibDeprecationWarning,
        message=r".*'labels' parameter of boxplot\(\) has been renamed.*",
    )
except Exception:
    pass


# -----------------------------
# Paths
# -----------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def default_data_dir() -> Path:
    return repo_root() / "data"


# -----------------------------
# Horizons (order + labels)
# -----------------------------
HORIZON_ORDER = ["r_post_pre", "r_5_post", "r_10_post", "r_20_post"]
HORIZON_LABEL = {
    "r_post_pre": "Reaction Next Day",
    "r_5_post": "Reaction After 5 Days",
    "r_10_post": "Reaction After 10 Days",
    "r_20_post": "Reaction After 20 Days",
}


# -----------------------------
# Color scheme (requested)
# -----------------------------
TICKER_COLOR = {
    "AAPL": "#A2AAAD",   # Apple grey
    "MSFT": "#000000",   # black
    "NVDA": "#76B900",   # NVIDIA green
    "AMZN": "#FF9900",   # Amazon orange
    "GOOG": "#34A853",   # Google green (not orange)
    "META": "#0866FF",   # Meta blue
    "TSLA": "#CC0000",   # Tesla red
}
CMAP_RDBU_GOODRED = "RdBu_r"  # red=good / blue=bad


# -----------------------------
# Feature families
# -----------------------------
BASE_FUNDAMENTALS = [
    "eps_surprise", "revenue_surprise",
    "eps_est", "eps_actual", "revenue_est", "revenue_actual",
]

BASE_NEWS = [
    "news_tminus1_pos_share","news_tminus1_neu_share","news_tminus1_neg_share",
    "news_tplus1_pos_share","news_tplus1_neu_share","news_tplus1_neg_share",
    "news_tminus1_k","news_tplus1_k",
    "d_news_pos","d_news_neg","d_news_neu",
]

BASE_TRANSCRIPT = [
    "transcript_pos_share","transcript_neu_share","transcript_neg_share",
    "transcript_k",
]


# -----------------------------
# Robust numeric prep
# -----------------------------
class CleanNumeric(BaseEstimator, TransformerMixin):
    """Force float64 and replace +/-inf with nan (so the imputer can handle it)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X[~np.isfinite(X)] = np.nan
        return X


class SafeStandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardize features: (X - mean) / std, but if std==0 set std=1.
    Avoids divide-by-zero warnings in older sklearn/matmul paths.
    """
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std = np.where(std > 0, std, 1.0)
        self.scale_ = std
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_


# -----------------------------
# Utilities
# -----------------------------
def _ticker_color(t: str) -> str:
    return TICKER_COLOR.get(t.upper(), "#444444")

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def savefig(fig, path: Path, bbox_inches=None, rect=None, tight=True):
    ensure_parent(path)
    if tight:
        try:
            if rect is not None:
                fig.tight_layout(rect=rect)
            else:
                fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, dpi=220, bbox_inches=bbox_inches)
    plt.close(fig)

def add_zero_lines(ax):
    ax.axhline(0.0, linewidth=1.1, alpha=0.6)
    ax.axvline(0.0, linewidth=1.1, alpha=0.6)

def label_horizon(h: str) -> str:
    return HORIZON_LABEL.get(h, h)

def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def print_step(msg: str):
    print(f"[STEP] {msg}", flush=True)

def print_ok(msg: str):
    print(f"[OK] {msg}", flush=True)

def print_warn(msg: str):
    print(f"[WARN] {msg}", flush=True)

def legend_bottom(fig, handles, labels, ncol=7, fontsize=9, y=-0.02):
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        bbox_to_anchor=(0.5, y),
    )

def boxplot_tick_kw(ax) -> str:
    """Matplotlib 3.9+: tick_labels. Older: labels."""
    try:
        sig = inspect.signature(ax.boxplot)
        return "tick_labels" if "tick_labels" in sig.parameters else "labels"
    except Exception:
        return "labels"


# -----------------------------
# Data loading + feature engineering
# -----------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime ordering
    if "t_earnings_date" in df.columns:
        df["t_earnings_date"] = pd.to_datetime(df["t_earnings_date"], errors="coerce")

    # Compute surprises if missing
    if "eps_surprise" not in df.columns and {"eps_actual", "eps_est"}.issubset(df.columns):
        est = pd.to_numeric(df["eps_est"], errors="coerce")
        act = pd.to_numeric(df["eps_actual"], errors="coerce")
        denom = np.where(np.abs(est.values) > 1e-12, np.abs(est.values), np.nan)
        df["eps_surprise"] = (act.values - est.values) / denom

    if "revenue_surprise" not in df.columns and {"revenue_actual", "revenue_est"}.issubset(df.columns):
        est = pd.to_numeric(df["revenue_est"], errors="coerce")
        act = pd.to_numeric(df["revenue_actual"], errors="coerce")
        denom = np.where(np.abs(est.values) > 1e-12, np.abs(est.values), np.nan)
        df["revenue_surprise"] = (act.values - est.values) / denom

    # News deltas (if base shares exist)
    if "d_news_pos" not in df.columns and {"news_tplus1_pos_share","news_tminus1_pos_share"}.issubset(df.columns):
        df["d_news_pos"] = pd.to_numeric(df["news_tplus1_pos_share"], errors="coerce") - pd.to_numeric(df["news_tminus1_pos_share"], errors="coerce")
    if "d_news_neg" not in df.columns and {"news_tplus1_neg_share","news_tminus1_neg_share"}.issubset(df.columns):
        df["d_news_neg"] = pd.to_numeric(df["news_tplus1_neg_share"], errors="coerce") - pd.to_numeric(df["news_tminus1_neg_share"], errors="coerce")
    if "d_news_neu" not in df.columns and {"news_tplus1_neu_share","news_tminus1_neu_share"}.issubset(df.columns):
        df["d_news_neu"] = pd.to_numeric(df["news_tplus1_neu_share"], errors="coerce") - pd.to_numeric(df["news_tminus1_neu_share"], errors="coerce")

    # Net sentiment conveniences
    if {"news_tminus1_pos_share","news_tminus1_neg_share"}.issubset(df.columns):
        df["news_pre_net"] = pd.to_numeric(df["news_tminus1_pos_share"], errors="coerce") - pd.to_numeric(df["news_tminus1_neg_share"], errors="coerce")
    if {"news_tplus1_pos_share","news_tplus1_neg_share"}.issubset(df.columns):
        df["news_post_net"] = pd.to_numeric(df["news_tplus1_pos_share"], errors="coerce") - pd.to_numeric(df["news_tplus1_neg_share"], errors="coerce")
    if {"news_pre_net","news_post_net"}.issubset(df.columns):
        df["news_d_net"] = pd.to_numeric(df["news_post_net"], errors="coerce") - pd.to_numeric(df["news_pre_net"], errors="coerce")

    if {"transcript_pos_share","transcript_neg_share"}.issubset(df.columns):
        df["tx_net"] = pd.to_numeric(df["transcript_pos_share"], errors="coerce") - pd.to_numeric(df["transcript_neg_share"], errors="coerce")

    # Cumulative-from-pre paths
    if {"r_post_pre","r_5_post"}.issubset(df.columns):
        df["r_pre_to_5"] = (1.0 + pd.to_numeric(df["r_post_pre"], errors="coerce")) * (1.0 + pd.to_numeric(df["r_5_post"], errors="coerce")) - 1.0
    if {"r_post_pre","r_10_post"}.issubset(df.columns):
        df["r_pre_to_10"] = (1.0 + pd.to_numeric(df["r_post_pre"], errors="coerce")) * (1.0 + pd.to_numeric(df["r_10_post"], errors="coerce")) - 1.0
    if {"r_post_pre","r_20_post"}.issubset(df.columns):
        df["r_pre_to_20"] = (1.0 + pd.to_numeric(df["r_post_pre"], errors="coerce")) * (1.0 + pd.to_numeric(df["r_20_post"], errors="coerce")) - 1.0

    return df


def load_event_panel(data_dir: Path, ticker: str) -> pd.DataFrame | None:
    p = data_dir / ticker / "events" / "event_panel_pilot.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["ticker"] = ticker
    df = add_engineered_features(df)
    if "t_earnings_date" in df.columns:
        df = df.sort_values("t_earnings_date").reset_index(drop=True)
    return df


# -----------------------------
# Models
# -----------------------------
def make_model_zoo():
    lin = Pipeline([
        ("clean", CleanNumeric()),
        ("imp", SimpleImputer(strategy="median")),
        ("vt", VarianceThreshold(0.0)),
        ("sc", SafeStandardScaler()),
        ("m", LinearRegression()),
    ])

    ridge = Pipeline([
        ("clean", CleanNumeric()),
        ("imp", SimpleImputer(strategy="median")),
        ("vt", VarianceThreshold(0.0)),
        ("sc", SafeStandardScaler()),
        ("m", Ridge(alpha=1.0)),
    ])

    rf = Pipeline([
        ("clean", CleanNumeric()),
        ("imp", SimpleImputer(strategy="median")),
        ("m", RandomForestRegressor(n_estimators=500, random_state=0)),
    ])

    gbrt = Pipeline([
        ("clean", CleanNumeric()),
        ("imp", SimpleImputer(strategy="median")),
        ("m", GradientBoostingRegressor(random_state=0)),
    ])

    return {
        "mean_baseline": None,
        "linear": lin,
        "ridge": ridge,
        "rf": rf,
        "gbrt": gbrt,
    }


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "dir_acc": np.nan, "r2": np.nan}

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    dir_acc = float(np.mean(np.sign(yt) == np.sign(yp)))

    if len(yt) < 3:
        r2 = np.nan
    else:
        try:
            r2 = float(r2_score(yt, yp))
        except Exception:
            r2 = np.nan

    return {"n": int(len(yt)), "mae": mae, "rmse": rmse, "dir_acc": dir_acc, "r2": r2}


# -----------------------------
# Evaluation modes
# -----------------------------
def eval_last4(df: pd.DataFrame, Xcols: list[str], ycol: str, model_name: str, model_obj, min_train: int):
    n = len(df)
    if n < (min_train + 4):
        return None  # not enough events

    train = df.iloc[: n - 4]
    test  = df.iloc[n - 4 :]

    y_tr = pd.to_numeric(train[ycol], errors="coerce").values
    y_te = pd.to_numeric(test[ycol], errors="coerce").values

    if model_name == "mean_baseline":
        mu = np.nanmean(y_tr)
        preds = np.full(len(y_te), mu, dtype=float)
        return preds, y_te

    X_tr = train[Xcols]
    X_te = test[Xcols]

    model_obj.fit(X_tr, y_tr)
    preds = model_obj.predict(X_te).astype(float)
    return preds, y_te


def eval_expanding(df: pd.DataFrame, Xcols: list[str], ycol: str, model_name: str, model_obj, min_train: int):
    n = len(df)
    if n <= min_train:
        return None

    y = pd.to_numeric(df[ycol], errors="coerce").values
    preds = np.full(n, np.nan, dtype=float)

    for i in range(min_train, n):
        tr = df.iloc[:i]
        te = df.iloc[i:i+1]

        y_tr = pd.to_numeric(tr[ycol], errors="coerce").values
        if model_name == "mean_baseline":
            preds[i] = float(np.nanmean(y_tr))
            continue

        X_tr = tr[Xcols]
        X_te = te[Xcols]
        model_obj.fit(X_tr, y_tr)
        preds[i] = float(model_obj.predict(X_te)[0])

    return preds, y


# -----------------------------
# Price + earnings-date plotting
# -----------------------------
def _read_prices_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # detect date column
    date_col = None
    for c in ["date", "Date", "datetime", "Datetime", "timestamp", "Timestamp"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    # prefer adj_close if present
    px_col = "adj_close" if "adj_close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else ("close" if "close" in df.columns else "Close"))
    if px_col not in df.columns:
        raise RuntimeError(f"Could not find close/adj_close in {path} columns={list(df.columns)}")

    df["px"] = pd.to_numeric(df[px_col], errors="coerce")
    return df[["date", "px"]].dropna().sort_values("date")


def _read_earnings_dates(data_dir: Path, ticker: str) -> pd.Series:
    p = data_dir / ticker / "calendar" / "earnings_calendar.csv"
    if not p.exists():
        return pd.Series([], dtype="datetime64[ns]")
    cal = pd.read_csv(p)
    d = pd.to_datetime(cal.get("earnings_date"), errors="coerce").dropna().sort_values()
    return d


def plot_price_with_earnings_dates(data_dir: Path, ticker: str, out_dir: Path):
    prices_path = data_dir / ticker / "prices" / "yf_ohlcv_daily.csv"
    if not prices_path.exists():
        print_warn(f"{ticker}: missing prices file {prices_path}")
        return

    px = _read_prices_csv(prices_path)
    ed = _read_earnings_dates(data_dir, ticker)

    fig = plt.figure(figsize=(12.5, 4.6))
    plt.plot(px["date"], px["px"], linewidth=2.0, alpha=0.92, color=_ticker_color(ticker))
    for d in ed:
        plt.axvline(d, linewidth=1.0, alpha=0.22)

    plt.title(f"{ticker}: Price with earnings dates (vertical lines)")
    plt.xlabel("Date")
    plt.ylabel("Adj Close (or Close)")
    savefig(fig, out_dir / f"price_with_earnings_{ticker}.png", bbox_inches="tight")
    print_ok(f"Wrote price_with_earnings_{ticker}.png")


def plot_mag7_price_lines(data_dir: Path, tickers: list[str], out_dir: Path, normalize=True):
    fig = plt.figure(figsize=(13.5, 5.6))

    for tkr in tickers:
        prices_path = data_dir / tkr / "prices" / "yf_ohlcv_daily.csv"
        if not prices_path.exists():
            continue
        px = _read_prices_csv(prices_path)
        y = px["px"].values
        if normalize and len(y) > 0 and np.isfinite(y[0]) and y[0] != 0:
            y = y / y[0]
        plt.plot(px["date"], y, linewidth=2.0, alpha=0.92, color=_ticker_color(tkr), label=tkr)

    plt.title("MAG7 price history (normalized)" if normalize else "MAG7 price history")
    plt.xlabel("Date")
    plt.ylabel("Normalized price" if normalize else "Price")
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_bottom(plt.gcf(), handles, labels, ncol=7, fontsize=9, y=-0.03)

    savefig(plt.gcf(), out_dir / ("mag7_prices_normalized.png" if normalize else "mag7_prices_raw.png"),
            bbox_inches="tight", rect=(0, 0.06, 1, 0.95))
    print_ok("Wrote MAG7 price lines plot")


# -----------------------------
# Plotting: performance heatmaps + rich visuals
# -----------------------------
def plot_model_suite_heatmaps(metrics_df: pd.DataFrame, out_dir: Path):
    print_step("Plotting model-suite heatmaps")

    base = metrics_df[metrics_df["model"] == "mean_baseline"][["ticker","horizon","mae"]].rename(columns={"mae":"mae_base"})
    suite = metrics_df[metrics_df["model"] != "mean_baseline"].merge(base, on=["ticker","horizon"], how="left")
    suite["mae_gain"] = suite["mae_base"] - suite["mae"]  # red=good

    suite_all = suite[suite["features"] == "all"].copy()
    if len(suite_all) == 0:
        print_warn("No rows for features=all in metrics; skipping model-suite heatmaps.")
        return

    agg = suite_all.groupby(["model","horizon"], as_index=False).agg(
        mae_gain=("mae_gain","mean"),
        dir_acc=("dir_acc","mean"),
        r2=("r2","mean"),
    )

    def make_piv(valcol):
        piv = agg.pivot_table(index="model", columns="horizon", values=valcol, aggfunc="mean").reindex(columns=HORIZON_ORDER)
        piv.columns = [HORIZON_LABEL[c] for c in piv.columns]
        return piv

    def heatmap(piv, title, fname, vmin=None, vmax=None):
        fig = plt.figure(figsize=(11, 4.2))
        plt.imshow(piv.values, aspect="auto", cmap=CMAP_RDBU_GOODRED, vmin=vmin, vmax=vmax)
        plt.xticks(range(len(piv.columns)), piv.columns, rotation=0)
        plt.yticks(range(len(piv.index)), piv.index)
        plt.colorbar()
        plt.title(title)
        savefig(fig, out_dir / fname, bbox_inches="tight")

    piv_gain = make_piv("mae_gain")
    vmax = np.nanmax(np.abs(piv_gain.values)) if np.isfinite(piv_gain.values).any() else 1.0
    heatmap(piv_gain, "MAE improvement vs baseline (ALL features, avg over tickers) | red=better", "suite_mae_gain.png", vmin=-vmax, vmax=vmax)

    piv_acc = make_piv("dir_acc")
    heatmap(piv_acc, "Directional accuracy (ALL features, avg over tickers) | red=better", "suite_dir_acc.png", vmin=0.0, vmax=1.0)

    piv_r2 = make_piv("r2")
    heatmap(piv_r2, "R² (ALL features, avg over tickers) | red=better", "suite_r2.png", vmin=-1.0, vmax=1.0)

    print_ok("Model-suite heatmaps written")


def plot_ablation_heatmap(metrics_df: pd.DataFrame, out_dir: Path):
    print_step("Plotting ablation heatmap (Ridge)")

    base = metrics_df[metrics_df["model"] == "mean_baseline"][["ticker","horizon","mae"]].rename(columns={"mae":"mae_base"})
    ridge = metrics_df[metrics_df["model"] == "ridge"].merge(base, on=["ticker","horizon"], how="left")
    ridge["mae_gain"] = ridge["mae_base"] - ridge["mae"]

    ridge = ridge[ridge["features"].isin(["fundamentals","news","transcript","all"])].copy()
    if len(ridge) == 0:
        print_warn("No ridge rows for ablation; skipping.")
        return

    piv = ridge.groupby(["features","horizon"], as_index=False)["mae_gain"].mean().pivot(index="features", columns="horizon", values="mae_gain").reindex(columns=HORIZON_ORDER)
    piv.columns = [HORIZON_LABEL[c] for c in piv.columns]
    vmax = np.nanmax(np.abs(piv.values)) if np.isfinite(piv.values).any() else 1.0

    fig = plt.figure(figsize=(11, 4.2))
    plt.imshow(piv.values, aspect="auto", cmap=CMAP_RDBU_GOODRED, vmin=-vmax, vmax=vmax)
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=0)
    plt.yticks(range(len(piv.index)), piv.index)
    plt.colorbar()
    plt.title("Ablation (Ridge): MAE improvement vs baseline | red=better")
    savefig(fig, out_dir / "ablation_ridge_mae_gain.png", bbox_inches="tight")
    print_ok("Ablation heatmap written")


def plot_scatter_matrix_horizons(df_all: pd.DataFrame, out_dir: Path):
    print_step("Plotting scatter-matrix of horizon interactions")

    hs = HORIZON_ORDER
    fig, axes = plt.subplots(len(hs), len(hs), figsize=(12.5, 12.5), sharex=False, sharey=False)

    for i, yi in enumerate(hs):
        for j, xj in enumerate(hs):
            ax = axes[i, j]
            if i == j:
                ax.axis("off")
                continue

            for tkr, g in df_all.groupby("ticker"):
                ax.scatter(
                    pd.to_numeric(g[xj], errors="coerce"),
                    pd.to_numeric(g[yi], errors="coerce"),
                    s=22,
                    alpha=0.78,
                    color=_ticker_color(tkr),
                    label=tkr if (i == 0 and j == 1) else None,
                )
            add_zero_lines(ax)

            if i == len(hs) - 1:
                ax.set_xlabel(HORIZON_LABEL[xj])
            if j == 0:
                ax.set_ylabel(HORIZON_LABEL[yi])

    handles, labels = axes[0, 1].get_legend_handles_labels()
    legend_bottom(fig, handles, labels, ncol=7, fontsize=9, y=-0.02)
    fig.suptitle("Return interactions across horizons (all events; colored by company)", y=0.98)

    savefig(fig, out_dir / "scatter_matrix_horizons.png", bbox_inches="tight", rect=(0, 0.06, 1, 0.95))
    print_ok("Scatter-matrix written")


def plot_event_lines_all_horizons(df_all: pd.DataFrame, out_dir: Path):
    print_step("Plotting event lines (2×2 horizons subplot)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()

    for ax, h in zip(axes, HORIZON_ORDER):
        for tkr, g in df_all.groupby("ticker"):
            gg = g.sort_values("t_earnings_date")
            ax.plot(
                gg["t_earnings_date"],
                pd.to_numeric(gg[h], errors="coerce"),
                marker="o",
                linewidth=2.0,
                alpha=0.92,
                color=_ticker_color(tkr),
                label=tkr,
            )
        ax.axhline(0.0, linewidth=1.1, alpha=0.6)
        ax.set_title(HORIZON_LABEL[h])
        ax.set_xlabel("Earnings date t")
        ax.set_ylabel("Return")

    handles, labels = axes[0].get_legend_handles_labels()
    legend_bottom(fig, handles, labels, ncol=7, fontsize=9, y=-0.02)
    fig.suptitle("Earnings-event returns over time (all horizons)", y=0.98)

    savefig(fig, out_dir / "event_lines_all_horizons.png", bbox_inches="tight", rect=(0, 0.06, 1, 0.95))
    print_ok("Event-line subplot written")


def plot_distributions_and_boxplots(df_all: pd.DataFrame, out_dir: Path):
    print_step("Plotting distributions + boxplots (subplots)")

    # Distributions (2x2)
    fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, h in zip(axes, HORIZON_ORDER):
        v = pd.to_numeric(df_all[h], errors="coerce").values
        v = v[np.isfinite(v)]
        ax.hist(v, bins=18, alpha=0.86)
        ax.axvline(0.0, linewidth=1.1, alpha=0.6)
        ax.set_title(HORIZON_LABEL[h])
        ax.set_xlabel("Return")
        ax.set_ylabel("Count")
    fig1.suptitle("Distributions of earnings-event returns (pooled across companies)", y=0.98)
    savefig(fig1, out_dir / "hist_returns_all_horizons.png", bbox_inches="tight")

    # Boxplots (2x2 horizons)
    tickers = sorted(df_all["ticker"].unique().tolist())
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    axes2 = axes2.ravel()
    kw = boxplot_tick_kw(axes2[0])

    for ax, h in zip(axes2, HORIZON_ORDER):
        data = []
        for tkr in tickers:
            vv = pd.to_numeric(df_all.loc[df_all["ticker"] == tkr, h], errors="coerce").values
            vv = vv[np.isfinite(vv)]
            data.append(vv)

        bp = ax.boxplot(data, showfliers=False, **{kw: tickers})
        for i, box in enumerate(bp["boxes"]):
            c = _ticker_color(tickers[i])
            box.set_color(c)
            try:
                box.set_facecolor("white")
            except Exception:
                pass
        for med in bp["medians"]:
            med.set_linewidth(2.2)
        ax.axhline(0.0, linewidth=1.1, alpha=0.6)
        ax.set_title(HORIZON_LABEL[h])
        ax.set_ylabel("Return")

    fig2.suptitle("Company comparison of returns (boxplots) across horizons", y=0.98)
    savefig(fig2, out_dir / "boxplots_all_horizons.png", bbox_inches="tight")

    print_ok("Distributions + boxplots written")


def plot_driver_correlation_and_scatter(df_all: pd.DataFrame, out_dir: Path):
    print_step("Plotting driver correlation + driver-vs-horizon scatters")

    drivers = [
        "eps_surprise", "revenue_surprise",
        "tx_net",
        "news_pre_net", "news_post_net", "news_d_net",
        "d_news_pos", "d_news_neg",
    ]
    cols = [c for c in drivers if c in df_all.columns] + HORIZON_ORDER

    X = df_all[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    corr = X.corr()

    # correlation heatmap
    fig = plt.figure(figsize=(12.5, 9.5))
    plt.imshow(corr.values, vmin=-1, vmax=1, cmap=CMAP_RDBU_GOODRED)
    plt.xticks(range(len(cols)), [HORIZON_LABEL.get(c, c) for c in cols], rotation=35, ha="right")
    plt.yticks(range(len(cols)), [HORIZON_LABEL.get(c, c) for c in cols])
    plt.colorbar()
    plt.title("Correlation heatmap: key drivers vs horizons (pooled events)")
    savefig(fig, out_dir / "corr_heatmap_drivers_targets.png", bbox_inches="tight")

    # driver vs horizon grid (4 horizons × 4 drivers => 16 panels)
    chosen = []
    for key in ["eps_surprise", "revenue_surprise", "tx_net", "news_d_net"]:
        if key in df_all.columns:
            chosen.append(key)
    if len(chosen) == 0:
        print_warn("No driver columns found for scatter grid; skipping.")
        return
    while len(chosen) < 4:
        chosen.append(chosen[-1])

    fig2, axes = plt.subplots(4, 4, figsize=(16, 14))
    for i, h in enumerate(HORIZON_ORDER):
        for j, drv in enumerate(chosen[:4]):
            ax = axes[i, j]
            for tkr, g in df_all.groupby("ticker"):
                ax.scatter(
                    pd.to_numeric(g[drv], errors="coerce"),
                    pd.to_numeric(g[h], errors="coerce"),
                    s=20,
                    alpha=0.75,
                    color=_ticker_color(tkr),
                    label=tkr if (i == 0 and j == 0) else None,
                )
            add_zero_lines(ax)
            ax.set_xlabel(drv)
            ax.set_ylabel(HORIZON_LABEL[h])
            ax.set_title(f"{drv} vs {HORIZON_LABEL[h]}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_bottom(fig2, handles, labels, ncol=7, fontsize=9, y=-0.02)
    fig2.suptitle("Driver vs horizon interactions (all events, colored by company)", y=0.98)
    savefig(fig2, out_dir / "driver_scatter_grid.png", bbox_inches="tight", rect=(0, 0.06, 1, 0.95))
    print_ok("Driver correlation + scatter grid written")


def plot_mean_cumulative_paths(df_all: pd.DataFrame, out_dir: Path):
    print_step("Plotting mean cumulative paths (from Pre)")

    needed = ["r_post_pre", "r_pre_to_5", "r_pre_to_10", "r_pre_to_20"]
    for c in needed:
        if c not in df_all.columns:
            print_warn(f"Missing {c}; skipping cumulative path plot.")
            return

    x = np.array([0, 1, 5, 10, 20], dtype=float)
    fig = plt.figure(figsize=(10.8, 5.4))

    for tkr, g in df_all.groupby("ticker"):
        y_post = pd.to_numeric(g["r_post_pre"], errors="coerce").mean()
        y5 = pd.to_numeric(g["r_pre_to_5"], errors="coerce").mean()
        y10 = pd.to_numeric(g["r_pre_to_10"], errors="coerce").mean()
        y20 = pd.to_numeric(g["r_pre_to_20"], errors="coerce").mean()
        y = np.array([0.0, y_post, y5, y10, y20], dtype=float)
        plt.plot(x, y, marker="o", linewidth=2.6, alpha=0.92, color=_ticker_color(tkr), label=tkr)

    plt.axhline(0.0, linewidth=1.1, alpha=0.6)
    plt.xticks(x, ["Pre", "Post", "+5", "+10", "+20"])
    plt.title("Mean cumulative return path around earnings (from Pre price)")
    plt.xlabel("Horizon")
    plt.ylabel("Mean cumulative return")

    handles, labels = plt.gca().get_legend_handles_labels()
    legend_bottom(plt.gcf(), handles, labels, ncol=7, fontsize=9, y=-0.03)
    savefig(plt.gcf(), out_dir / "mean_cumulative_path_by_company.png", bbox_inches="tight", rect=(0, 0.06, 1, 0.95))
    print_ok("Mean cumulative path plot written")


def plot_event_paths_transparent_plus_mean(df_all: pd.DataFrame, out_dir: Path):
    """
    For each ticker: plot all event paths (faint) + mean path (bold).
    Path: Pre (0), Post (r_post_pre), +5 (r_pre_to_5), +10, +20.
    """
    print_step("Plotting event paths (all events transparent + mean bold)")

    needed = ["r_post_pre", "r_pre_to_5", "r_pre_to_10", "r_pre_to_20"]
    for c in needed:
        if c not in df_all.columns:
            print_warn(f"Missing {c}; skipping event-path plot.")
            return

    tickers = sorted(df_all["ticker"].unique().tolist())
    rows = 2
    cols = int(np.ceil(len(tickers) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.3 * rows))
    axes = np.array(axes).ravel()

    x = np.array([0, 1, 5, 10, 20], dtype=float)
    xlab = ["Pre", "Post", "+5", "+10", "+20"]

    for ax, tkr in zip(axes, tickers):
        g = df_all[df_all["ticker"] == tkr].sort_values("t_earnings_date")
        c = _ticker_color(tkr)

        # all event paths (faint)
        for _, r in g.iterrows():
            y = np.array([
                0.0,
                float(pd.to_numeric(r["r_post_pre"], errors="coerce")),
                float(pd.to_numeric(r["r_pre_to_5"], errors="coerce")),
                float(pd.to_numeric(r["r_pre_to_10"], errors="coerce")),
                float(pd.to_numeric(r["r_pre_to_20"], errors="coerce")),
            ], dtype=float)
            if np.all(np.isfinite(y)):
                ax.plot(x, y, linewidth=1.5, alpha=0.12, color=c)

        # mean (bold)
        mean_y = np.array([
            0.0,
            pd.to_numeric(g["r_post_pre"], errors="coerce").mean(),
            pd.to_numeric(g["r_pre_to_5"], errors="coerce").mean(),
            pd.to_numeric(g["r_pre_to_10"], errors="coerce").mean(),
            pd.to_numeric(g["r_pre_to_20"], errors="coerce").mean(),
        ], dtype=float)
        ax.plot(x, mean_y, linewidth=3.2, alpha=0.95, color=c)

        ax.axhline(0.0, linewidth=1.0, alpha=0.55)
        ax.set_title(f"{tkr}: paths (thin=events, bold=mean)")
        ax.set_xticks(x)
        ax.set_xticklabels(xlab)
        ax.set_ylabel("Cumulative return from Pre")

    for k in range(len(tickers), len(axes)):
        axes[k].axis("off")

    fig.suptitle("Earnings-event cumulative paths (all events transparent + mean bold)", y=0.98)
    savefig(fig, out_dir / "event_paths_transparent_plus_mean.png", bbox_inches="tight")
    print_ok("Event paths figure written")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL","MSFT","AMZN","GOOG","META","NVDA","TSLA"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-subdir", default="reports/pilot_mag7/full_run")
    ap.add_argument("--eval-mode", choices=["last4","expanding"], default="last4")
    ap.add_argument("--min-train", type=int, default=8)
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    out_dir = data_dir / args.out_subdir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print_step(f"Loading event panels for tickers={args.tickers}")
    frames = []
    for tkr in args.tickers:
        df = load_event_panel(data_dir, tkr)
        if df is None:
            print_warn(f"Missing event panel for {tkr} (data/{tkr}/events/event_panel_pilot.csv). Skipping.")
            continue

        missing = [h for h in HORIZON_ORDER if h not in df.columns]
        if missing:
            print_warn(f"{tkr}: missing horizon cols {missing}; skipping ticker.")
            continue

        frames.append(df)

    if not frames:
        raise RuntimeError("No usable event panels found. Build event_panel_pilot.csv first.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sort_values(["ticker","t_earnings_date"]).reset_index(drop=True)

    df_all.to_csv(out_dir / "events_pooled_mag7.csv", index=False)
    print_ok(f"Wrote pooled events: {out_dir/'events_pooled_mag7.csv'}")

    zoo = make_model_zoo()

    print_step(f"Running evaluation: mode={args.eval_mode} (min_train={args.min_train})")
    all_rows = []

    for tkr, g in df_all.groupby("ticker"):
        g = g.sort_values("t_earnings_date").reset_index(drop=True)

        fam_cols = {
            "fundamentals": safe_cols(g, BASE_FUNDAMENTALS),
            "news": safe_cols(g, BASE_NEWS),
            "transcript": safe_cols(g, BASE_TRANSCRIPT),
            "all": sorted(set(safe_cols(g, BASE_FUNDAMENTALS) + safe_cols(g, BASE_NEWS) + safe_cols(g, BASE_TRANSCRIPT))),
        }

        for horizon in HORIZON_ORDER:
            for features_name, cols in fam_cols.items():
                if not cols:
                    continue

                for model_name, model_obj in zoo.items():
                    if model_name == "mean_baseline":
                        cols_use = []
                    else:
                        cols_use = cols

                    if args.eval_mode == "last4":
                        res = eval_last4(g, cols_use, horizon, model_name, model_obj, args.min_train)
                        if res is None:
                            continue
                        preds, y_true = res
                        m = compute_metrics(y_true, preds)
                    else:
                        res = eval_expanding(g, cols_use, horizon, model_name, model_obj, args.min_train)
                        if res is None:
                            continue
                        preds, y_true = res
                        m = compute_metrics(y_true, preds)

                    all_rows.append({
                        "ticker": tkr,
                        "eval_mode": args.eval_mode,
                        "model": model_name,
                        "features": ("none" if model_name == "mean_baseline" else features_name),
                        "horizon": horizon,
                        "horizon_label": HORIZON_LABEL[horizon],
                        **m,
                    })

        print_ok(f"Evaluated {tkr} ({len(g)} events)")

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = out_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print_ok(f"Wrote metrics: {metrics_path}")

    # -----------------------------
    # VISUALS
    # -----------------------------
    print_step("Generating visualization pack")

    # Price plots (requested)
    print_step("Plotting price-with-earnings per ticker")
    for tkr in args.tickers:
        plot_price_with_earnings_dates(data_dir, tkr, fig_dir)

    print_step("Plotting MAG7 combined price lines")
    plot_mag7_price_lines(data_dir, args.tickers, fig_dir, normalize=True)

    # Performance heatmaps
    plot_model_suite_heatmaps(metrics_df, fig_dir)
    plot_ablation_heatmap(metrics_df, fig_dir)

    # Pooled visuals from df_all
    plot_scatter_matrix_horizons(df_all, fig_dir)
    plot_event_lines_all_horizons(df_all, fig_dir)
    plot_distributions_and_boxplots(df_all, fig_dir)
    plot_mean_cumulative_paths(df_all, fig_dir)
    plot_event_paths_transparent_plus_mean(df_all, fig_dir)
    plot_driver_correlation_and_scatter(df_all, fig_dir)

    print_ok(f"All done. Figures in: {fig_dir}")
    print_ok("Top files to open first:")
    print("  - mag7_prices_normalized.png")
    print("  - price_with_earnings_NVDA.png (and others)")
    print("  - suite_mae_gain.png")
    print("  - suite_dir_acc.png")
    print("  - suite_r2.png")
    print("  - ablation_ridge_mae_gain.png")
    print("  - scatter_matrix_horizons.png")
    print("  - event_paths_transparent_plus_mean.png")
    print("  - event_lines_all_horizons.png")
    print("  - driver_scatter_grid.png")


if __name__ == "__main__":
    main()
