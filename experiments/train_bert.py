# -*- coding: utf-8 -*-
"""Fine-tune bert-base-chinese for the same binary sentiment task and report test metrics."""

import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = r"E:\python\sentiment-analysis"
MODEL_NAME = "bert-base-chinese"
MAX_LEN = 128
BATCH = 32
EPOCHS = 6
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, flush=True)


class DS(Dataset):
    def __init__(self, path, tokenizer, max_len):
        import pandas as pd
        self.df = pd.read_csv(path, encoding="utf-8")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        text = str(self.df.iloc[i]["text"])
        label = int(self.df.iloc[i]["label"])
        enc = self.tokenizer(text, max_length=self.max_len, padding="max_length",
                             truncation=True, return_tensors="pt")
        return enc["input_ids"][0], enc["attention_mask"][0], torch.tensor(label)


tok = AutoTokenizer.from_pretrained(MODEL_NAME)
train_ds = DS(os.path.join(ROOT, "data", "train.csv"), tok, MAX_LEN)
val_ds = DS(os.path.join(ROOT, "data", "val.csv"), tok, MAX_LEN)
test_ds = DS(os.path.join(ROOT, "data", "test.csv"), tok, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)
test_loader = DataLoader(test_ds, batch_size=BATCH)
print("data sizes:", len(train_ds), len(val_ds), len(test_ds), flush=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def run_epoch(loader, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()
    preds, gold = [], []
    with torch.set_grad_enabled(train_mode):
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            if train_mode:
                optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask).logits
            loss = torch.nn.functional.cross_entropy(out, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            gold.extend(labels.cpu().numpy())
    return accuracy_score(gold, preds)


best_val = 0
save_path = os.path.join(ROOT, "models", "best_model_bert.pth")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_acc = run_epoch(train_loader, True)
    va_acc = run_epoch(val_loader, False)
    print(f"Epoch {epoch} | train {tr_acc:.4f} | val {va_acc:.4f} | {time.time() - t0:.0f}s", flush=True)
    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), save_path)

model.load_state_dict(torch.load(save_path, map_location=DEVICE))
model.eval()
preds, gold = [], []
with torch.no_grad():
    for ids, mask, labels in test_loader:
        out = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits
        preds.extend(torch.argmax(out, dim=1).cpu().numpy())
        gold.extend(labels.numpy())
print("test accuracy:", round(accuracy_score(gold, preds), 4), flush=True)
print(classification_report(gold, preds, target_names=["负面", "正面"], digits=4))
print("saved:", save_path, flush=True)
