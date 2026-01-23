#!/usr/bin/env python3
"""
Train FinBERT + LoRA on Financial PhraseBank (Apple Silicon / MPS).
- Uses Transformers v5-style eval_strategy
- Float32 (no AMP) for MPS stability
- Uses safetensors
- Optimizer: adamw_torch
- Fixes label mapping: PhraseBank -> FinBERT label space
"""

import os
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# Device & env
# ---------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
print("Using device:", device)

# ---------------------------
# Load CSV splits produced by 01_get_phrasebank_*.py
# ---------------------------
def as_split(path: str):
    return load_dataset("csv", data_files=path)["train"]

train = as_split("data/interim/train.csv")
val   = as_split("data/interim/val.csv")
test  = as_split("data/interim/test.csv")
ds = DatasetDict(train=train, validation=val, test=test)

# Ensure ints
def to_int(x):
    x["label"] = int(x["label"])
    return x
ds = ds.map(to_int)

# ---------------------------
# Remap labels (CRITICAL)
# PhraseBank: 0=neg, 1=neu, 2=pos
# FinBERT expects ids: positive=0, negative=1, neutral=2
# Mapping: 0(neg)->1, 1(neu)->2, 2(pos)->0
# ---------------------------
def remap(ex):
    m = {0: 1, 1: 2, 2: 0}
    ex["label"] = m[int(ex["label"])]
    return ex

ds = ds.map(remap)

# ---------------------------
# Tokenizer & preprocessing
# ---------------------------
model_id = "ProsusAI/finbert"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

def preprocess(batch):
    return tok(batch["text"], truncation=True, max_length=256)

cols_to_drop = [c for c in ds["train"].column_names if c not in ("text", "label")]
ds_tok = ds.map(preprocess, batched=True, remove_columns=cols_to_drop)
ds_tok = ds_tok.rename_column("label", "labels")

collator = DataCollatorWithPadding(tokenizer=tok)

# ---------------------------
# Model (float32, safetensors) + LoRA
# ---------------------------
label2id = {"positive": 0, "negative": 1, "neutral": 2}
id2label = {v: k for k, v in label2id.items()}

mdl = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=3,
    use_safetensors=True,
    low_cpu_mem_usage=True,
    label2id=label2id,
    id2label=id2label,
)

lcfg = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["query", "key", "value", "output.dense"],
    bias="none",
    task_type="SEQ_CLS",
)
mdl = get_peft_model(mdl, lcfg)
mdl.to(device)

# ---------------------------
# Metrics
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }

# ---------------------------
# Training args (Transformers v5: eval_strategy)
# ---------------------------
out_dir = "models/finbert_lora"
Path(out_dir).mkdir(parents=True, exist_ok=True)

args = TrainingArguments(
    output_dir=out_dir,
    num_train_epochs=3,                # real run
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    eval_strategy="epoch",             # <- per your request, v5 keyword
    save_strategy="epoch",
    logging_steps=25,
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    fp16=False,                        # no AMP on MPS
    bf16=False,
    dataloader_pin_memory=False,       # avoid MPS pin_memory warning
    optim="adamw_torch",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=42,
)

trainer = Trainer(
    model=mdl,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    processing_class=tok,              # future-proof replacement for tokenizer=
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(out_dir)
tok.save_pretrained(out_dir)

print("Training complete. Model saved to", out_dir)