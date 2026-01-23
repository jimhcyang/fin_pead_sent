#!/usr/bin/env python3
from datasets import load_dataset
from pathlib import Path
import pandas as pd

out = Path("data/interim")
out.mkdir(parents=True, exist_ok=True)

raw = load_dataset(
    "financial_phrasebank",
    "sentences_allagree",
    trust_remote_code=True
)["train"]

# 80/10/10 split
tmp = raw.train_test_split(test_size=0.2, seed=42)
tv  = tmp["test"].train_test_split(test_size=0.5, seed=42)
train, val, test = tmp["train"], tv["train"], tv["test"]

def to_csv(ds, path):
    df = pd.DataFrame({"text": ds["sentence"], "label": ds["label"]})
    df.to_csv(path, index=False)

to_csv(train, "data/interim/train.csv")
to_csv(val,   "data/interim/val.csv")
to_csv(test,  "data/interim/test.csv")
print("Saved train/val/test to data/interim/")