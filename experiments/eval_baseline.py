# -*- coding: utf-8 -*-
"""Evaluate the current best_model.pth on the test set (real baseline metrics)."""

import os
import sys

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT = r"E:\python\sentiment-analysis"
sys.path.insert(0, ROOT)

from dataset import SentimentDataset
from model import BiLSTM_Attention
from vocab import load_vocab

DEVICE = torch.device("cpu")
MAX_LEN = 100

vocab = load_vocab(os.path.join(ROOT, "data", "vocab.pkl"))
print("vocab size:", len(vocab))

test_ds = SentimentDataset(os.path.join(ROOT, "data", "test.csv"), vocab, MAX_LEN)
test_loader = DataLoader(test_ds, batch_size=256)
print("test size:", len(test_ds))

model = BiLSTM_Attention(len(vocab), 300, 128, 2)
model.load_state_dict(torch.load(os.path.join(ROOT, "models", "best_model.pth"), map_location="cpu"))
model.eval()

preds, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        out = model(x)
        preds.extend(torch.argmax(out, dim=1).numpy())
        labels.extend(y.numpy())

print("accuracy:", round(accuracy_score(labels, preds), 4))
print(classification_report(labels, preds, target_names=["负面", "正面"], digits=4))
print("confusion matrix:\n", confusion_matrix(labels, preds))
