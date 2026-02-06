#!/usr/bin/env python3
# scripts/_lm.py
"""
Loughran–McDonald (LM) Master Dictionary utilities.

What this module does:
- Ensures a local cached copy of the LM Master Dictionary CSV exists.
- Loads sentiment category word sets (NEG/POS/UNC/LIT/CONSTR/STRONG/WEAK).
- Scores text using LM-style filtering:
    * tokenize -> UPPERCASE
    * keep only tokens that are in the LM Master Dictionary core word list
    * drop LM stopwords (LM's slightly modified list from the official sample code)
  Denominators are based on the remaining LM-valid word tokens.

Environment overrides:
- LM_DICT_PATH: absolute path to a local CSV
- LM_DICT_URL : direct-download URL to a CSV

Notes:
- The Notre Dame webpage sometimes changes or blocks direct downloads.
- The default URL below uses the current Google Drive "usercontent" direct-download link.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests

from _text_utils import tokenize


# --- Default direct-download URL (CSV) from the Notre Dame LM page (Google Drive mirror) ---
# Source page: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
DEFAULT_LM_URL = (
    "https://drive.usercontent.google.com/u/0/uc"
    "?id=1cfg_w3USlRFS97wo7XQmYnuzhpmzboAY&export=download"
)

# Column name variants across LM releases / file formats
CANON_COLS = {
    "negative": ["negative", "neg"],
    "positive": ["positive", "pos"],
    "uncertainty": ["uncertainty", "uncertain"],
    "litigious": ["litigious", "litigation"],
    "constraining": ["constraining", "constraint", "constraints"],
    "strong_modal": ["strong_modal", "strongmodal", "modal_strong", "strongmodalwords"],
    "weak_modal": ["weak_modal", "weakmodal", "modal_weak", "weakmodalwords"],
    "complexity": ["complexity", "complex"],
}

# LM sample stopwords (slightly modified vs. common lists)
LM_STOPWORDS: Set[str] = {
    'ME','MY','MYSELF','WE','OUR','OURS','OURSELVES','YOU','YOUR','YOURS','YOURSELF','YOURSELVES',
    'HE','HIM','HIS','HIMSELF','SHE','HER','HERS','HERSELF','IT','ITS','ITSELF','THEY','THEM',
    'THEIR','THEIRS','THEMSELVES','WHAT','WHICH','WHO','WHOM','THIS','THAT','THESE','THOSE',
    'AM','IS','ARE','WAS','WERE','BE','BEEN','BEING','HAVE','HAS','HAD','HAVING','DO','DOES',
    'DID','DOING','AN','THE','AND','BUT','IF','OR','BECAUSE','AS','UNTIL','WHILE','OF','AT','BY',
    'FOR','WITH','ABOUT','BETWEEN','INTO','THROUGH','DURING','BEFORE','AFTER','ABOVE','BELOW',
    'TO','FROM','UP','DOWN','IN','OUT','ON','OFF','OVER','UNDER','AGAIN','FURTHER','THEN','ONCE',
    'HERE','THERE','WHEN','WHERE','WHY','HOW','ALL','ANY','BOTH','EACH','FEW','MORE','MOST',
    'OTHER','SOME','SUCH','NO','NOR','NOT','ONLY','OWN','SAME','SO','THAN','TOO','VERY','CAN',
    'JUST','SHOULD','NOW','AMONG'
}


def _pick_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        key = str(n).strip().lower()
        if key in cols:
            return cols[key]
    return None


def _looks_like_csv(bytes_: bytes) -> bool:
    """
    Guard against HTML downloads from blocked endpoints.
    """
    head = bytes_[:5000].decode("utf-8", errors="ignore").strip().lower()
    if head.startswith("<!doctype html") or head.startswith("<html"):
        return False
    # LM CSV almost always contains a header with "word" and "negative" somewhere near top
    return ("word" in head.splitlines()[0]) and ("negative" in head or "positive" in head)


def ensure_lm_dictionary(data_dir: str = "data") -> Path:
    """
    Returns a local path to the LM Master Dictionary CSV.

    Priority:
      1) env LM_DICT_PATH (existing file)
      2) data/_resources/lm/LoughranMcDonald_MasterDictionary.csv (cache)
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
        r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
        r.raise_for_status()
        if not _looks_like_csv(r.content):
            raise RuntimeError("Download did not look like a CSV (may be HTML / blocked).")
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
            "3) Or set LM_DICT_URL to a reachable *direct-download* CSV URL\n\n"
            f"Original error: {e}"
        )


def load_lm_sets(data_dir: str = "data") -> Dict[str, Set[str]]:
    """
    Loads LM dictionary and returns a dict of sets.

    Keys:
      - "master": all LM dictionary words (uppercase)
      - "stopwords": LM stopwords (uppercase)
      - sentiment categories: negative/positive/uncertainty/litigious/constraining/strong_modal/weak_modal/complexity
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
    out["master"] = set(df[word_col].tolist())
    out["stopwords"] = set(LM_STOPWORDS)

    for canon, variants in CANON_COLS.items():
        col = _pick_col(df, variants)
        if col is None:
            out[canon] = set()
            continue
        # LM files typically store the year included; negative values indicate removal.
        # Membership: > 0
        s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        out[canon] = set(df.loc[s > 0, word_col].tolist())

    return out


def _lm_filter_tokens(toks: List[str], lm_sets: Dict[str, Set[str]]) -> List[str]:
    master = lm_sets.get("master", set())
    stop = lm_sets.get("stopwords", set())
    out = []
    for t in toks:
        tt = (t or "").upper().strip()
        if len(tt) < 2:
            continue
        if tt in stop:
            continue
        if master and tt not in master:
            continue
        out.append(tt)
    return out


def score_lm(text: str, lm_sets: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Returns LM category counts and proportions.

    Denominator = number of LM-valid word tokens:
      token in MasterDictionary AND not in LM stopword list.
    """
    toks_raw = tokenize(text)
    toks = _lm_filter_tokens(toks_raw, lm_sets)

    n_raw = float(len(toks_raw))
    n = float(len(toks))
    if n <= 0:
        return {
            "n_tokens_raw": n_raw,
            "n_tokens": 0.0,
            "lm_pos": 0.0, "lm_neg": 0.0, "lm_unc": 0.0, "lm_lit": 0.0,
            "lm_constr": 0.0, "lm_strong": 0.0, "lm_weak": 0.0, "lm_complex": 0.0,
            "lm_pos_prop": 0.0, "lm_neg_prop": 0.0, "lm_unc_prop": 0.0, "lm_lit_prop": 0.0,
            "lm_constr_prop": 0.0,
            "lm_tone": 0.0,
        }

    pos_set = lm_sets.get("positive", set())
    neg_set = lm_sets.get("negative", set())
    unc_set = lm_sets.get("uncertainty", set())
    lit_set = lm_sets.get("litigious", set())
    con_set = lm_sets.get("constraining", set())
    st_set = lm_sets.get("strong_modal", set())
    wk_set = lm_sets.get("weak_modal", set())
    cx_set = lm_sets.get("complexity", set())

    counts = {
        "lm_pos": sum(1 for t in toks if t in pos_set),
        "lm_neg": sum(1 for t in toks if t in neg_set),
        "lm_unc": sum(1 for t in toks if t in unc_set),
        "lm_lit": sum(1 for t in toks if t in lit_set),
        "lm_constr": sum(1 for t in toks if t in con_set),
        "lm_strong": sum(1 for t in toks if t in st_set),
        "lm_weak": sum(1 for t in toks if t in wk_set),
        "lm_complex": sum(1 for t in toks if t in cx_set),
    }

    pos = float(counts["lm_pos"])
    neg = float(counts["lm_neg"])

    return {
        "n_tokens_raw": n_raw,
        "n_tokens": n,
        **{k: float(v) for k, v in counts.items()},
        "lm_pos_prop": pos / n,
        "lm_neg_prop": neg / n,
        "lm_unc_prop": float(counts["lm_unc"]) / n,
        "lm_lit_prop": float(counts["lm_lit"]) / n,
        "lm_constr_prop": float(counts["lm_constr"]) / n,
        "lm_tone": (pos - neg) / n,
    }
