#!/usr/bin/env python3
# scripts/_lm.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests

from _text_utils import tokenize


# Try Notre Dame first; if blocked, user can set LM_DICT_URL or LM_DICT_PATH.
DEFAULT_LM_URL = "https://sraf.nd.edu/wp-content/uploads/2014/11/LoughranMcDonald_MasterDictionary_1993-2023.csv"

CANON_COLS = {
    "negative": ["negative", "neg"],
    "positive": ["positive", "pos"],
    "uncertainty": ["uncertainty", "uncertain"],
    "litigious": ["litigious", "litigation"],
    "constraining": ["constraining", "constraint"],
    "strong_modal": ["strong_modal", "strongmodal", "modal_strong"],
    "weak_modal": ["weak_modal", "weakmodal", "modal_weak"],
}


def _pick_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None


def ensure_lm_dictionary(data_dir: str = "data") -> Path:
    """
    Returns a local path to the LM Master Dictionary CSV.

    Priority:
      1) env LM_DICT_PATH (existing file)
      2) data/_resources/lm/LoughranMcDonald_MasterDictionary.csv
      3) download from env LM_DICT_URL or DEFAULT_LM_URL
    """
    env_path = os.getenv("LM_DICT_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    out_dir = Path(data_dir) / "_resources" / "lm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "LoughranMcDonald_MasterDictionary.csv"
    if out_path.exists():
        return out_path

    url = os.getenv("LM_DICT_URL", DEFAULT_LM_URL)
    headers = {"User-Agent": "Mozilla/5.0"}  # helps with some 403 setups
    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return out_path
    except Exception as e:
        raise RuntimeError(
            "Could not obtain LM dictionary automatically.\n"
            f"- Tried URL: {url}\n"
            f"- Expected local cache: {out_path}\n\n"
            "Fix options:\n"
            "1) Download the LM Master Dictionary CSV manually and save it to:\n"
            f"   {out_path}\n"
            "2) Or set environment variable LM_DICT_PATH=/path/to/the.csv\n"
            "3) Or set LM_DICT_URL to a reachable CSV URL\n\n"
            f"Original error: {e}"
        )


def load_lm_sets(data_dir: str = "data") -> Dict[str, Set[str]]:
    """
    Loads LM dictionary and returns category -> set(words).
    """
    p = ensure_lm_dictionary(data_dir=data_dir)
    df = pd.read_csv(p)

    # detect word column
    word_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"word", "words", "term"}:
            word_col = c
            break
    if word_col is None:
        word_col = df.columns[0]  # fallback

    df[word_col] = df[word_col].astype(str).str.upper().str.strip()

    out: Dict[str, Set[str]] = {}
    for canon, variants in CANON_COLS.items():
        col = _pick_col(df, variants)
        if col is None:
            out[canon] = set()
            continue
        # Many LM files use 0/1; coerce truthy
        s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        out[canon] = set(df.loc[s > 0, word_col].tolist())

    return out


def score_lm(text: str, lm_sets: Dict[str, Set[str]]) -> Dict[str, float]:
    toks = tokenize(text)
    n = float(len(toks))
    if n <= 0:
        return {
            "n_tokens": 0.0,
            "lm_pos": 0.0, "lm_neg": 0.0, "lm_unc": 0.0, "lm_lit": 0.0,
            "lm_constr": 0.0, "lm_strong": 0.0, "lm_weak": 0.0,
            "lm_pos_prop": 0.0, "lm_neg_prop": 0.0, "lm_unc_prop": 0.0, "lm_lit_prop": 0.0,
            "lm_tone": 0.0,
        }

    counts = {
        "lm_pos": sum(1 for t in toks if t in lm_sets.get("positive", set())),
        "lm_neg": sum(1 for t in toks if t in lm_sets.get("negative", set())),
        "lm_unc": sum(1 for t in toks if t in lm_sets.get("uncertainty", set())),
        "lm_lit": sum(1 for t in toks if t in lm_sets.get("litigious", set())),
        "lm_constr": sum(1 for t in toks if t in lm_sets.get("constraining", set())),
        "lm_strong": sum(1 for t in toks if t in lm_sets.get("strong_modal", set())),
        "lm_weak": sum(1 for t in toks if t in lm_sets.get("weak_modal", set())),
    }

    pos = float(counts["lm_pos"])
    neg = float(counts["lm_neg"])

    return {
        "n_tokens": n,
        **{k: float(v) for k, v in counts.items()},
        "lm_pos_prop": pos / n,
        "lm_neg_prop": neg / n,
        "lm_unc_prop": float(counts["lm_unc"]) / n,
        "lm_lit_prop": float(counts["lm_lit"]) / n,
        "lm_tone": (pos - neg) / n,
    }
