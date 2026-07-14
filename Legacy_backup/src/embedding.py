# -*- coding: utf-8 -*-
"""
Word2Vec 词嵌入脚本 - 训练词向量模型
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from config import (
    TRAIN_DATA_PATH, WORD2VEC_MODEL_PATH, MAX_SEQ_LEN,
    W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_WORKERS
)
import os


def train_word2vec():
    """训练 Word2Vec 模型"""
    print("=" * 50)
    print("训练 Word2Vec 模型...")
    print("=" * 50)

    # 加载训练数据
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    # 将分词后的文本转换为词列表
    sentences = [text.split() for text in train_df['cut_text']]

    print(f"训练语料数量: {len(sentences)} 条")

    # 训练 Word2Vec
    model = Word2Vec(
        sentences,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=W2V_WORKERS,
        sg=1  # 使用 Skip-gram 算法
    )

    # 保存模型
    model.save(WORD2VEC_MODEL_PATH)
    print(f"Word2Vec 模型已保存: {WORD2VEC_MODEL_PATH}")
    print(f"词汇表大小: {len(model.wv.key_to_index)}")
    print(f"词向量维度: {model.wv.vector_size}")

    return model


def text_to_matrix(text, w2v_model, max_len=MAX_SEQ_LEN):
    """
    将文本转换为词向量矩阵
    返回: (max_len, embedding_dim) 的 numpy 数组
    """
    words = text.split()
    matrix = []

    for word in words:
        if word in w2v_model.wv:
            matrix.append(w2v_model.wv[word])
        else:
            # 未登录词用零向量填充
            matrix.append(np.zeros(W2V_VECTOR_SIZE))

    # Padding 或截断
    if len(matrix) < max_len:
        # 填充零向量
        matrix += [np.zeros(W2V_VECTOR_SIZE)] * (max_len - len(matrix))
    else:
        # 截断
        matrix = matrix[:max_len]

    return np.array(matrix, dtype=np.float32)


def load_data_as_vectors(data_path, w2v_model):
    """加载 CSV 数据并转换为向量形式"""
    df = pd.read_csv(data_path)
    X = np.array([text_to_matrix(text, w2v_model) for text in df['cut_text']])
    y = df['label'].values
    return X, y


def main():
    train_word2vec()
    print("\n✅ Word2Vec 训练完成！")


if __name__ == '__main__':
    main()