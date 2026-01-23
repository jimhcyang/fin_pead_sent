#!/usr/bin/env python3
"""
Evaluate FinBERT+LoRA on the PhraseBank test split.
- Uses tokenizer __call__ with padding=True (faster with fast tokenizers)
- Applies the same label remap used in training (PB -> FinBERT space)
- MPS-friendly
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models/finbert_lora"
TEST_CSV = "data/interim/test.csv"
BATCH = 64  # adjust if needed

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_grad_enabled(False)
print("Using device:", device)

# Load model/tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
mdl.eval().to(device)

# Load test set (PhraseBank label space)
test = load_dataset("csv", data_files=TEST_CSV)["train"]

def to_int(x):
    x["label"] = int(x["label"])
    return x

test = test.map(to_int)

# Remap gold labels to FinBERT id space (PB -> FIN): {0:neg,1:neu,2:pos} -> {0:pos,1:neg,2:neu}
PB_TO_FIN = {0: 1, 1: 2, 2: 0}
def remap_label(ex):
    ex["label"] = PB_TO_FIN[int(ex["label"])]
    return ex

test = test.map(remap_label)

# Predict using tokenizer __call__ with padding
preds, gold = [], []
for i in range(0, len(test), BATCH):
    batch = test[i : i + BATCH]
    texts = batch["text"]
    labels = [int(x) for x in batch["label"]]

    inputs = tok(
        texts,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = mdl(**inputs)

    preds.extend(out.logits.argmax(-1).detach().cpu().tolist())
    gold.extend(labels)

# Class names from model config (FinBERT space)
id2label = mdl.config.id2label
target_names = [id2label[i] for i in range(mdl.config.num_labels)]

report_txt = classification_report(gold, preds, target_names=target_names, digits=4)
cm = confusion_matrix(gold, preds).tolist()

print(report_txt)

Path("data/interim").mkdir(parents=True, exist_ok=True)
with open("data/interim/test_report.json", "w") as f:
    json.dump(
        {
            "classification_report_text": report_txt,
            "confusion_matrix": cm,
            "target_names": target_names,
        },
        f,
        indent=2,
    )
print("Saved metrics to data/interim/test_report.json")