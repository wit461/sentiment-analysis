# -*- coding: utf-8 -*-
"""Train BiLSTM+Attention with pretrained 300-dim embeddings and report test metrics."""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

ROOT = r"E:\python\sentiment-analysis"
sys.path.insert(0, ROOT)

from dataset import SentimentDataset
from model import BiLSTM_Attention
from vocab import load_vocab

EMBEDDING_DIM = 300
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 40
LR = 0.001
PATIENCE = 5
MAX_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, flush=True)

vocab = load_vocab(os.path.join(ROOT, "data", "vocab.pkl"))
print("vocab size:", len(vocab), flush=True)


def build_emb_matrix(vocab, vec_path, dim=300):
    emb = np.random.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    hit = 0
    t0 = time.time()
    with open(vec_path, "r", encoding="utf-8") as f:
        f.readline()  # 头部信息
        for line in f:
            word, _, vec_str = line.partition(" ")
            idx = vocab.get(word)
            if idx is None:
                continue
            emb[idx] = np.fromstring(vec_str, sep=" ", dtype=np.float32)
            hit += 1
    print(f"emb scan done in {time.time() - t0:.0f}s, hits: {hit}/{len(vocab)}", flush=True)
    return torch.tensor(emb, dtype=torch.float)


pretrained = build_emb_matrix(vocab, os.path.join(ROOT, "data", "cc.zh.300.vec"))

train_ds = SentimentDataset(os.path.join(ROOT, "data", "train.csv"), vocab, MAX_LEN)
val_ds = SentimentDataset(os.path.join(ROOT, "data", "val.csv"), vocab, MAX_LEN)
test_ds = SentimentDataset(os.path.join(ROOT, "data", "test.csv"), vocab, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

labels = np.array([int(y) for _, y in train_ds])
n = len(labels)
w0 = n / (2 * (labels == 0).sum())
w1 = n / (2 * (labels == 1).sum())
criterion = nn.CrossEntropyLoss(weight=torch.tensor([w0, w1], dtype=torch.float).to(DEVICE))

model = BiLSTM_Attention(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, 2,
                         pretrained_embeddings=pretrained).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)


def run_epoch(loader, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()
    total_loss, preds, gold = 0, [], []
    with torch.set_grad_enabled(train_mode):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train_mode:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            gold.extend(y.cpu().numpy())
    return total_loss / len(gold), accuracy_score(gold, preds)


best_val = 0
patience = 0
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
save_path = os.path.join(ROOT, "models", "best_model_emb.pth")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, True)
    va_loss, va_acc = run_epoch(val_loader, False)
    scheduler.step(va_loss)
    print(f"Epoch {epoch:2d} | train {tr_acc:.4f} | val {va_acc:.4f} | {time.time() - t0:.1f}s", flush=True)
    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), save_path)
        patience = 0
    else:
        patience += 1
        if patience >= PATIENCE:
            print("early stop", flush=True)
            break

model.load_state_dict(torch.load(save_path))
te_loss, te_acc = run_epoch(test_loader, False)
print("test accuracy:", round(te_acc, 4), flush=True)

preds, gold = [], []
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        preds.extend(torch.argmax(model(x.to(DEVICE)), dim=1).cpu().numpy())
        gold.extend(y.numpy())
print(classification_report(gold, preds, target_names=["负面", "正面"], digits=4))
print("saved:", save_path, flush=True)
