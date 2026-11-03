# -*- coding: utf-8 -*-
"""Soft-voting ensemble of BERT and RoBERTa checkpoints on the test set."""

import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = r"E:\python\sentiment-analysis"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH = 32


def build(name, weights):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    return tok, model


tok_b, model_b = build("bert-base-chinese", os.path.join(ROOT, "models", "best_model_bert.pth"))
tok_r, model_r = build("hfl/chinese-roberta-wwm-ext",
                       os.path.join(ROOT, "models", "best_model_roberta.pth"))

df = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), encoding="utf-8")
texts = [str(t) for t in df["text"].tolist()]
gold = [int(v) for v in df["label"].tolist()]

enc_b = tok_b(texts, max_length=MAX_LEN, padding=True, truncation=True, return_tensors="pt")
enc_r = tok_r(texts, max_length=MAX_LEN, padding=True, truncation=True, return_tensors="pt")

preds = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH):
        ids_b = enc_b["input_ids"][i:i + BATCH].to(DEVICE)
        mask_b = enc_b["attention_mask"][i:i + BATCH].to(DEVICE)
        ids_r = enc_r["input_ids"][i:i + BATCH].to(DEVICE)
        mask_r = enc_r["attention_mask"][i:i + BATCH].to(DEVICE)
        logits_b = model_b(input_ids=ids_b, attention_mask=mask_b).logits
        logits_r = model_r(input_ids=ids_r, attention_mask=mask_r).logits
        probs = torch.softmax(logits_b, dim=1) + torch.softmax(logits_r, dim=1)
        preds.extend(torch.argmax(probs, dim=1).cpu().numpy().tolist())

print("ensemble accuracy:", round(accuracy_score(gold, preds), 4), flush=True)
print(classification_report(gold, preds, target_names=["负面", "正面"], digits=4))
