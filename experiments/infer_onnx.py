# -*- coding: utf-8 -*-
"""ONNX Runtime inference with the quantized BERT model (single + batch)."""

import os
import sys
import time

import numpy as np

ROOT = r"E:\python\sentiment-analysis"
sys.path.insert(0, ROOT)

import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_PATH = os.path.join(ROOT, "models", "best_model_bert_quant.onnx")
MAX_LEN = 128

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
tok = AutoTokenizer.from_pretrained("bert-base-chinese")


def predict_single(text):
    enc = tok(str(text), max_length=MAX_LEN, padding="max_length",
              truncation=True, return_tensors="np")
    logits = sess.run(None, {"input_ids": enc["input_ids"],
                             "attention_mask": enc["attention_mask"],
                             "token_type_ids": enc["token_type_ids"]})[0]
    prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    pred = int(np.argmax(prob))
    return pred, float(prob[0][pred])


def predict_batch(texts):
    enc = tok([str(t) for t in texts], max_length=MAX_LEN, padding=True,
              truncation=True, return_tensors="np")
    logits = sess.run(None, {"input_ids": enc["input_ids"],
                             "attention_mask": enc["attention_mask"],
                             "token_type_ids": enc["token_type_ids"]})[0]
    prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    preds = np.argmax(prob, axis=1).tolist()
    confs = [float(prob[i][p]) for i, p in enumerate(preds)]
    return preds, confs


if __name__ == "__main__":
    sample = "酒店位置不错，服务态度很好，房间也很干净"
    predict_single(sample)
    t0 = time.perf_counter()
    for _ in range(30):
        predict_single(sample)
    print(f"single: {(time.perf_counter() - t0) / 30 * 1000:.1f} ms", flush=True)

    texts = [sample] * 64
    predict_batch(texts)
    t0 = time.perf_counter()
    for _ in range(5):
        predict_batch(texts)
    per = (time.perf_counter() - t0) / (5 * len(texts)) * 1000
    print(f"batch per item: {per:.2f} ms", flush=True)

    p, c = predict_single("隔音太差，很吵")
    print("sample:", "负面" if p == 0 else "正面", round(c, 4), flush=True)
