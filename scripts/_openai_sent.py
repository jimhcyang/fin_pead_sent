#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI sentiment helper with a prompt that mimics FinBERT/LM outputs.

Returns:
  - oa_pos, oa_neg, oa_neu probabilities (sum to 1.0)
  - oa_tone = oa_pos - oa_neg

Environment:
  OPENAI_API_KEY (required)
  OPENAI_MODEL (default: gpt-4o-mini)
"""

from __future__ import annotations

import os
from typing import List, Dict
import json

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None


PROMPT = """You are a financial sentiment rater. Read the text delimited by <text>.
Return ONLY a compact JSON object with three probabilities that sum to 1.0:
{"pos": ..., "neg": ..., "neu": ...}
Use a style similar to FinBERT/LM: 'pos' > 0.55 if clearly positive, 'neg' > 0.55 if clearly negative; otherwise push weight to 'neu'.
Tone = pos - neg (will be computed downstream). No explanations, just JSON."""


def _client():
    if openai is None:
        raise ImportError("openai package not installed. pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = api_key
    return openai


def score_openai_batch(texts: List[str], model: str | None = None) -> List[Dict[str, float]]:
    if not texts:
        return []

    cli = _client()
    mname = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    out: List[Dict[str, float]] = []
    for tx in texts:
        msg = [{"role": "system", "content": PROMPT}, {"role": "user", "content": f"<text>\n{tx}\n</text>"}]
        resp = cli.ChatCompletion.create(model=mname, messages=msg, max_tokens=50, temperature=0.0)
        content = resp["choices"][0]["message"]["content"]
        try:
            data = json.loads(content)
            pos = float(data.get("pos", 0.0))
            neg = float(data.get("neg", 0.0))
            neu = float(data.get("neu", 0.0))
        except Exception:
            # Fallback: all neutral
            pos = neg = 0.0
            neu = 1.0
        total = pos + neg + neu
        if total > 0:
            pos, neg, neu = pos / total, neg / total, neu / total
        out.append(
            {
                "oa_pos": pos,
                "oa_neg": neg,
                "oa_neu": neu,
                "oa_tone": pos - neg,
            }
        )
    return out
