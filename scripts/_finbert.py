#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT scoring helper.

Uses HuggingFace transformers to load a sequence classification model (default: ProsusAI/finbert).
Returns per-text probabilities for POS / NEG / NEU plus a simple tone = pos - neg.

Environment overrides:
  FINBERT_MODEL_NAME  : HF model id or local path (default: ProsusAI/finbert)
  FINBERT_MAX_LENGTH  : max token length (default: 256)
  FINBERT_BATCH_SIZE  : batch size (default: 32)
"""

from __future__ import annotations

import os
from typing import List, Dict, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_finbert(model_name: str | None = None) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, str]:
    name = model_name or os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    dev = _device()
    mdl.eval().to(dev)
    return tok, mdl, dev


def _softmax(logits: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()


def score_finbert_batch(
    texts: List[str],
    tok: AutoTokenizer,
    mdl: AutoModelForSequenceClassification,
    device: str,
    max_length: int | None = None,
) -> List[Dict[str, float]]:
    if not texts:
        return []

    max_len = int(max_length or os.getenv("FINBERT_MAX_LENGTH", 256))
    enc = tok(
        texts,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = mdl(**enc)
    probs = _softmax(out.logits)

    # Expect labels like {0: 'negative', 1: 'neutral', 2: 'positive'}
    id2label = mdl.config.id2label or {0: "negative", 1: "neutral", 2: "positive"}
    label_map = {v.lower(): k for k, v in id2label.items()}
    idx_neg = label_map.get("negative", 0)
    idx_neu = label_map.get("neutral", 1)
    idx_pos = label_map.get("positive", 2)

    out_rows: List[Dict[str, float]] = []
    for pv in probs:
        pos = float(pv[idx_pos])
        neg = float(pv[idx_neg])
        neu = float(pv[idx_neu])
        out_rows.append(
            {
                "finbert_pos": pos,
                "finbert_neg": neg,
                "finbert_neu": neu,
                "finbert_tone": pos - neg,
            }
        )
    return out_rows


def chunked(iterable: List[str], n: int) -> List[List[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

