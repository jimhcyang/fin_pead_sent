#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()


def ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=w, adjust=False, min_periods=w).mean()


def rsi(close: pd.Series, w: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / w, adjust=False, min_periods=w).mean()
    avg_loss = loss.ewm(alpha=1 / w, adjust=False, min_periods=w).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, w: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = true_range(high, low, prev_close)
    return tr.rolling(w, min_periods=w).mean()


def bollinger(close: pd.Series, w: int = 20, n_std: float = 2.0):
    mid = sma(close, w)
    sd = close.rolling(w, min_periods=w).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    width = (upper - lower) / (mid.replace(0, np.nan))
    return mid, upper, lower, width


def obv(adj_close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(adj_close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, w: int = 20) -> pd.Series:
    # Money flow multiplier
    denom = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / denom
    mfv = mfm * volume
    return mfv.rolling(w, min_periods=w).sum() / volume.rolling(w, min_periods=w).sum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, w: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    pos = rmf.where(tp.diff() > 0, 0.0)
    neg = rmf.where(tp.diff() < 0, 0.0).abs()
    pos_sum = pos.rolling(w, min_periods=w).sum()
    neg_sum = neg.rolling(w, min_periods=w).sum()
    mfr = pos_sum / (neg_sum + 1e-12)
    return 100 - (100 / (1 + mfr))
