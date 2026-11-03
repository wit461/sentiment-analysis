# -*- coding: utf-8 -*-
"""EDA-style augmentation for the minority (negative) class: random delete / insert / swap."""

import os
import random

import pandas as pd

ROOT = r"E:\python\sentiment-analysis"
SEED = 42

rng = random.Random(SEED)

df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), encoding="utf-8")
neg = df[df["label"] == 0].copy()
pos = df[df["label"] == 1].copy()
print("before: neg", len(neg), "pos", len(pos), flush=True)


def augment(text):
    # 字符级增强：直接作用于原始中文文本，兼容 BERT 输入
    chars = list(str(text))
    if len(chars) < 10:
        return None
    choice = rng.choice(["delete", "insert", "swap"])
    if choice == "delete":
        k = max(1, int(len(chars) * 0.08))
        for _ in range(k):
            del chars[rng.randrange(len(chars))]
    elif choice == "insert":
        for _ in range(2):
            w = rng.choice(chars)
            chars.insert(rng.randrange(len(chars) + 1), w)
    else:
        i, j = rng.randrange(len(chars)), rng.randrange(len(chars))
        chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


rows = []
for _, row in neg.iterrows():
    aug_text = augment(row["text"])
    if aug_text:
        rows.append({"text": aug_text, "label": 0, "cut": ""})

aug_df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
print("generated:", len(aug_df), flush=True)

# 目标：负面样本扩到与正面相当（约 1:1），但不超采样过多
target_neg = len(pos)
if len(aug_df) > target_neg - len(neg):
    aug_df = aug_df.sample(n=target_neg - len(neg), random_state=SEED)
new_df = pd.concat([df, aug_df], ignore_index=True)
print("after: neg", (new_df["label"] == 0).sum(), "pos", (new_df["label"] == 1).sum(), flush=True)
new_df.to_csv(os.path.join(ROOT, "data", "train_aug.csv"), index=False, encoding="utf-8")
print("saved data/train_aug.csv", flush=True)
