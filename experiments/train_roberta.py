# -*- coding: utf-8 -*-
"""Fine-tune hfl/chinese-roberta-wwm-ext with warmup + cosine schedule + class weights."""

import os
import sys
import time

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

ROOT = r"E:\python\sentiment-analysis"
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MAX_LEN = 128
BATCH = 32
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, flush=True)

torch.manual_seed(SEED)


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

labels = [int(y) for _, _, y in train_ds]
n_neg = labels.count(0)
n_pos = labels.count(1)
w0 = len(labels) / (2 * n_neg)
w1 = len(labels) / (2 * n_pos)
print(f"class weights: neg={w0:.3f} pos={w1:.3f}", flush=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped = [
    {"params": [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optimizer_grouped, lr=LR)
total_steps = EPOCHS * len(train_loader)
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
criterion = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([w0, w1], dtype=torch.float).to(DEVICE))


def run_epoch(loader, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()
    preds, gold = [], []
    with torch.set_grad_enabled(train_mode):
        for ids, mask, ys in loader:
            ids, mask, ys = ids.to(DEVICE), mask.to(DEVICE), ys.to(DEVICE)
            if train_mode:
                optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask).logits
            loss = criterion(out, ys)
            if train_mode:
                loss.backward()
                optimizer.step()
                scheduler.step()
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            gold.extend(ys.cpu().numpy())
    return accuracy_score(gold, preds)


best_val = 0
save_path = os.path.join(ROOT, "models", "best_model_roberta.pth")
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
    for ids, mask, ys in test_loader:
        out = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits
        preds.extend(torch.argmax(out, dim=1).cpu().numpy())
        gold.extend(ys.numpy())
print("test accuracy:", round(accuracy_score(gold, preds), 4), flush=True)
print(classification_report(gold, preds, target_names=["负面", "正面"], digits=4))
print("saved:", save_path, flush=True)
