#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F


# ---------------------------
# Paths (repo_root/scripts/*.py)
# ---------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    return repo_root() / "data"


# ---------------------------
# Utilities
# ---------------------------

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fmt_ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def asof_trading_date(d: date, trading_dates: List[date]) -> Optional[date]:
    """Return latest trading date <= d, or None if none exists."""
    # trading_dates is sorted ascending
    import bisect
    i = bisect.bisect_right(trading_dates, d) - 1
    if i < 0:
        return None
    return trading_dates[i]


def shift_trading_date(d: date, n: int, date_to_idx: Dict[date, int], trading_dates: List[date]) -> Optional[date]:
    """Shift by n trading days using the trading calendar from price data."""
    if d not in date_to_idx:
        return None
    j = date_to_idx[d] + n
    if j < 0 or j >= len(trading_dates):
        return None
    return trading_dates[j]


def closest_transcript_date(t: date, available: List[date], max_abs_days: int = 2) -> Optional[date]:
    if not available:
        return None
    best = None
    best_abs = 10**9
    for d in available:
        abs_days = abs((d - t).days)
        if abs_days < best_abs:
            best = d
            best_abs = abs_days
    if best is None or best_abs > max_abs_days:
        return None
    return best


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    cleaned = text.replace("\r\n", " ").replace("\n", " ").strip()
    parts = SENT_SPLIT_RE.split(cleaned)
    # light cleanup
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 5:
            continue
        out.append(p)
    return out


# ---------------------------
# FinBERT(+LoRA) inference
# ---------------------------

@dataclass
class InferConfig:
    model_dir: Path
    batch_size: int = 64
    max_length: int = 256


class FinbertInferer:
    """
    Uses whatever is at model_dir (FinBERT or FinBERT+LoRA adapter saved as a full model dir).
    Expects id2label includes {positive, negative, neutral} (case-insensitive).
    """

    def __init__(self, cfg: InferConfig) -> None:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(device)
        torch.set_grad_enabled(False)

        self.model_dir = Path(cfg.model_dir)
        self.batch_size = int(cfg.batch_size)
        self.max_length = int(cfg.max_length)

        print(f"[INFO] Loading sentiment model from: {self.model_dir}")
        print(f"[INFO] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval().to(self.device)

        # normalize id2label/label2id
        id2label_raw = self.model.config.id2label
        self.id2label = {int(k): str(v).lower() for k, v in id2label_raw.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}

        for lbl in ["positive", "negative", "neutral"]:
            if lbl not in self.label2id:
                raise RuntimeError(f"Model labels missing '{lbl}'. id2label={self.id2label}")

    def predict_labels(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        labels: List[str] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            probs = F.softmax(out.logits, dim=-1).detach().cpu().numpy()

            for pv in probs:
                pred_id = int(np.argmax(pv))
                labels.append(self.id2label.get(pred_id, str(pred_id)))
        return labels


def counts_pos_neu_neg(labels: List[str]) -> Tuple[int, int, int]:
    pos = sum(1 for x in labels if x == "positive")
    neg = sum(1 for x in labels if x == "negative")
    neu = sum(1 for x in labels if x == "neutral")
    return pos, neu, neg


def shares(pos: int, neu: int, neg: int) -> Tuple[float, float, float]:
    k = pos + neu + neg
    if k <= 0:
        return (float("nan"), float("nan"), float("nan"))
    return (pos / k, neu / k, neg / k)


# ---------------------------
# Loaders
# ---------------------------

def load_prices(data_dir: Path, ticker: str) -> pd.DataFrame:
    p = data_dir / ticker / "prices" / "yf_ohlcv_daily.csv"
    if not p.exists():
        raise RuntimeError(f"Missing prices file: {p} (run 01_yf_prices.py)")

    df = pd.read_csv(p)
    if "date" not in df.columns or "adj_close" not in df.columns:
        raise RuntimeError(f"Unexpected columns in {p}: {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_calendar(data_dir: Path, ticker: str) -> pd.DataFrame:
    p = data_dir / ticker / "calendar" / "earnings_calendar.csv"
    if not p.exists():
        raise RuntimeError(f"Missing calendar file: {p} (run 03_fmp_earnings_calendar.py)")
    df = pd.read_csv(p)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    df["announce_timing"] = df["announce_timing"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["earnings_date"]).sort_values("earnings_date").reset_index(drop=True)
    return df


def list_transcript_dates(data_dir: Path, ticker: str) -> List[date]:
    tdir = data_dir / ticker / "transcripts"
    if not tdir.exists():
        return []
    out: List[date] = []
    for child in tdir.iterdir():
        if not child.is_dir():
            continue
        try:
            out.append(parse_ymd(child.name))
        except Exception:
            continue
    out.sort()
    return out


def read_transcript_text(data_dir: Path, ticker: str, tdate: date) -> Optional[str]:
    p = data_dir / ticker / "transcripts" / fmt_ymd(tdate) / "transcript.txt"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="ignore")


def read_news_jsonl(data_dir: Path, ticker: str, day: date) -> List[Dict[str, Any]]:
    p = data_dir / ticker / "news" / f"news_{fmt_ymd(day)}.jsonl"
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def news_texts(items: List[Dict[str, Any]], max_chars: int = 2000) -> List[str]:
    out = []
    for x in items:
        title = str(x.get("title") or "").strip()
        text = str(x.get("text") or "").strip()
        blob = (title + ". " + text).strip() if title else text
        blob = re.sub(r"\s+", " ", blob).strip()
        if not blob:
            continue
        if len(blob) > max_chars:
            blob = blob[:max_chars]
        out.append(blob)
    return out


# ---------------------------
# Panel builder
# ---------------------------

def build_event_panel(
    data_dir: Path,
    ticker: str,
    inferer: FinbertInferer,
    max_transcript_sentences: int = 1500,
    transcript_match_window_days: int = 2,
) -> pd.DataFrame:
    cal = load_calendar(data_dir, ticker)
    px = load_prices(data_dir, ticker)

    trading_dates = px["date"].tolist()
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    adj_map = {d: float(v) for d, v in zip(px["date"], px["adj_close"])}

    tx_dates = list_transcript_dates(data_dir, ticker)

    rows = []
    for _, r in cal.iterrows():
        t = r["earnings_date"]
        timing = str(r.get("announce_timing") or "").lower().strip()

        # Anchor to trading calendar
        t_asof = asof_trading_date(t, trading_dates)
        if t_asof is None:
            continue

        # Pre/post rule (your definition)
        if timing == "bmo":
            post_date = t_asof
            pre_date = shift_trading_date(post_date, -1, date_to_idx, trading_dates)
        else:
            # treat unknown as amc
            pre_date = t_asof
            post_date = shift_trading_date(pre_date, 1, date_to_idx, trading_dates)

        if pre_date is None or post_date is None:
            continue

        # PEAD horizons from post_date
        d5 = shift_trading_date(post_date, 5, date_to_idx, trading_dates)
        d10 = shift_trading_date(post_date, 10, date_to_idx, trading_dates)
        d20 = shift_trading_date(post_date, 20, date_to_idx, trading_dates)

        pre_px = adj_map.get(pre_date, float("nan"))
        post_px = adj_map.get(post_date, float("nan"))
        px5 = adj_map.get(d5, float("nan")) if d5 else float("nan")
        px10 = adj_map.get(d10, float("nan")) if d10 else float("nan")
        px20 = adj_map.get(d20, float("nan")) if d20 else float("nan")

        # Returns/diffs
        def ret(a, b) -> float:
            if not (np.isfinite(a) and np.isfinite(b)) or a == 0:
                return float("nan")
            return b / a - 1.0

        post_minus_pre = post_px - pre_px if np.isfinite(post_px) and np.isfinite(pre_px) else float("nan")
        r_post_pre = ret(pre_px, post_px)

        drift5 = px5 - post_px if np.isfinite(px5) and np.isfinite(post_px) else float("nan")
        drift10 = px10 - post_px if np.isfinite(px10) and np.isfinite(post_px) else float("nan")
        drift20 = px20 - post_px if np.isfinite(px20) and np.isfinite(post_px) else float("nan")

        r_5_post = ret(post_px, px5)
        r_10_post = ret(post_px, px10)
        r_20_post = ret(post_px, px20)

        # Fundamentals
        eps_est = safe_float(r.get("eps_est"))
        eps_actual = safe_float(r.get("eps_actual"))
        rev_est = safe_float(r.get("revenue_est"))
        rev_actual = safe_float(r.get("revenue_actual"))

        eps_surprise = (eps_actual - eps_est) / abs(eps_est) if np.isfinite(eps_actual) and np.isfinite(eps_est) and eps_est != 0 else float("nan")
        rev_surprise = (rev_actual - rev_est) / abs(rev_est) if np.isfinite(rev_actual) and np.isfinite(rev_est) and rev_est != 0 else float("nan")

        # News days (calendar t-1, t+1)
        news_d_minus = t - timedelta(days=1)
        news_d_plus = t + timedelta(days=1)

        news_m1_items = read_news_jsonl(data_dir, ticker, news_d_minus)
        news_p1_items = read_news_jsonl(data_dir, ticker, news_d_plus)

        news_m1_labels = inferer.predict_labels(news_texts(news_m1_items))
        news_p1_labels = inferer.predict_labels(news_texts(news_p1_items))

        nm1_pos, nm1_neu, nm1_neg = counts_pos_neu_neg(news_m1_labels)
        np1_pos, np1_neu, np1_neg = counts_pos_neu_neg(news_p1_labels)

        nm1_pos_s, nm1_neu_s, nm1_neg_s = shares(nm1_pos, nm1_neu, nm1_neg)
        np1_pos_s, np1_neu_s, np1_neg_s = shares(np1_pos, np1_neu, np1_neg)

        # Transcript date matching (closest transcript folder to earnings_date)
        tx_date = closest_transcript_date(t, tx_dates, max_abs_days=transcript_match_window_days)
        tx_text = read_transcript_text(data_dir, ticker, tx_date) if tx_date else None

        if tx_text:
            sents = split_sentences(tx_text)
            if max_transcript_sentences is not None and len(sents) > max_transcript_sentences:
                sents = sents[:max_transcript_sentences]
            tx_labels = inferer.predict_labels(sents)
            tx_pos, tx_neu, tx_neg = counts_pos_neu_neg(tx_labels)
            tx_pos_s, tx_neu_s, tx_neg_s = shares(tx_pos, tx_neu, tx_neg)
            tx_k = tx_pos + tx_neu + tx_neg
        else:
            tx_pos = tx_neu = tx_neg = 0
            tx_pos_s = tx_neu_s = tx_neg_s = float("nan")
            tx_k = 0

        rows.append(
            {
                "ticker": ticker,
                "t_earnings_date": fmt_ymd(t),
                "announce_timing": timing,

                "pre_date": fmt_ymd(pre_date),
                "post_date": fmt_ymd(post_date),
                "d5": fmt_ymd(d5) if d5 else None,
                "d10": fmt_ymd(d10) if d10 else None,
                "d20": fmt_ymd(d20) if d20 else None,

                "pre_adj_close": pre_px,
                "post_adj_close": post_px,
                "adj_close_5": px5,
                "adj_close_10": px10,
                "adj_close_20": px20,

                "post_minus_pre": post_minus_pre,
                "r_post_pre": r_post_pre,
                "drift_5": drift5,
                "drift_10": drift10,
                "drift_20": drift20,
                "r_5_post": r_5_post,
                "r_10_post": r_10_post,
                "r_20_post": r_20_post,

                "eps_est": eps_est,
                "eps_actual": eps_actual,
                "revenue_est": rev_est,
                "revenue_actual": rev_actual,
                "eps_surprise": eps_surprise,
                "revenue_surprise": rev_surprise,

                "news_tminus1_date": fmt_ymd(news_d_minus),
                "news_tplus1_date": fmt_ymd(news_d_plus),

                "news_tminus1_k": (nm1_pos + nm1_neu + nm1_neg),
                "news_tminus1_pos": nm1_pos,
                "news_tminus1_neu": nm1_neu,
                "news_tminus1_neg": nm1_neg,
                "news_tminus1_pos_share": nm1_pos_s,
                "news_tminus1_neu_share": nm1_neu_s,
                "news_tminus1_neg_share": nm1_neg_s,

                "news_tplus1_k": (np1_pos + np1_neu + np1_neg),
                "news_tplus1_pos": np1_pos,
                "news_tplus1_neu": np1_neu,
                "news_tplus1_neg": np1_neg,
                "news_tplus1_pos_share": np1_pos_s,
                "news_tplus1_neu_share": np1_neu_s,
                "news_tplus1_neg_share": np1_neg_s,

                "transcript_date_used": fmt_ymd(tx_date) if tx_date else None,
                "transcript_sentence_k": tx_k,
                "transcript_pos": tx_pos,
                "transcript_neu": tx_neu,
                "transcript_neg": tx_neg,
                "transcript_pos_share": tx_pos_s,
                "transcript_neu_share": tx_neu_s,
                "transcript_neg_share": tx_neg_s,
            }
        )

    out = pd.DataFrame(rows)
    return out


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model-dir", default=None, help="Default: repo_root/bert/models/finbert_lora")
    ap.add_argument("--max-transcript-sentences", type=int, default=1500)
    ap.add_argument("--transcript-match-window-days", type=int, default=2)
    ap.add_argument("--out-name", default="event_panel_pilot")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()

    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else (repo_root() / "bert" / "models" / "finbert_lora")
    )

    inferer = FinbertInferer(InferConfig(model_dir=model_dir))

    df = build_event_panel(
        data_dir=data_dir,
        ticker=ticker,
        inferer=inferer,
        max_transcript_sentences=args.max_transcript_sentences,
        transcript_match_window_days=args.transcript_match_window_days,
    )

    out_dir = data_dir / ticker / "events"
    ensure_dir(out_dir)

    out_csv = out_dir / f"{args.out_name}.csv"
    df.to_csv(out_csv, index=False)

    meta = {
        "ticker": ticker,
        "rows": int(df.shape[0]),
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "max_transcript_sentences": args.max_transcript_sentences,
        "transcript_match_window_days": args.transcript_match_window_days,
        "created_at_local": datetime.now().isoformat(),
        "notes": [
            "Pre/Post dates follow: amc => pre=t, post=next trading day; bmo => pre=prev trading day, post=t.",
            "PEAD horizons are 5/10/20 trading days after post_date using the trading calendar from yf_ohlcv_daily.csv.",
            "News days are calendar t-1 and t+1 (ET-day files produced by 05_fmp_news.py).",
            "Transcript date is the closest transcript folder date within ±window days of earnings_date.",
        ],
    }
    (out_dir / f"{args.out_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Wrote: {out_csv} ({df.shape[0]} rows)")
    print(f"[OK] Wrote: {out_dir / f'{args.out_name}.meta.json'}")


if __name__ == "__main__":
    main()
