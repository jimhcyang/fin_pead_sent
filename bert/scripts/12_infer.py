#!/usr/bin/env python3
"""
Batch inference for FinBERT+LoRA without transformers.pipeline
- Uses tokenizer __call__ with padding=True
- Direct model forward on MPS/CPU/CUDA
- Outputs label strings from model.config.id2label + softmax confidence
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def softmax_logits(logits: torch.Tensor) -> np.ndarray:
    # logits: [B, C] -> probs: [B, C]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models/finbert_lora")
    p.add_argument("--text", help="Single text to classify")
    p.add_argument("--input_csv", help="CSV with a 'text' column")
    p.add_argument("--text_col", default="text")
    p.add_argument("--out_csv", default="data/predictions/preds.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=256)
    args = p.parse_args()

    # Device
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    print("Using device:", device)

    # Load model / tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    mdl.eval().to(device)
    id2label = mdl.config.id2label

    rows = []

    # Case 1: single text
    if args.text:
        inputs = tok(
            [args.text],
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl(**inputs)
        probs = softmax_logits(out.logits)[0]
        pred_id = int(np.argmax(probs))
        pred_label = id2label[pred_id] if id2label else str(pred_id)
        rows.append([args.text, pred_label, float(probs[pred_id])])

    # Case 2: CSV batch
    elif args.input_csv:
        df = pd.read_csv(args.input_csv)
        if args.text_col not in df.columns:
            raise ValueError(f"Column '{args.text_col}' not in {args.input_csv}.")
        texts = df[args.text_col].astype(str).tolist()

        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i : i + args.batch_size]
            inputs = tok(
                batch_texts,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = mdl(**inputs)
            probs = softmax_logits(out.logits)
            pred_ids = probs.argmax(axis=-1)

            for t, pid, pvec in zip(batch_texts, pred_ids, probs):
                label = id2label[int(pid)] if id2label else str(int(pid))
                rows.append([t, label, float(pvec[int(pid)])])

    # Write output (if any)
    if rows:
        Path(os.path.dirname(args.out_csv) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "pred_label", "confidence"])
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.out_csv}")
    else:
        print("Nothing to do. Pass --text or --input_csv.")

if __name__ == "__main__":
    main()