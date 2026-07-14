# -*- coding: utf-8 -*-
"""
模型评估脚本 - 在测试集上评估模型性能
"""

import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_model
from embedding import load_data_as_vectors
from config import (
    TEST_DATA_PATH, LSTM_MODEL_PATH, WORD2VEC_MODEL_PATH,
    EMBEDDING_DIM, LSTM_HIDDEN_SIZE, NUM_CLASSES,
    LSTM_NUM_LAYERS, IDX_TO_LABEL
)
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def evaluate():
    """评估模型性能"""
    print("=" * 60)
    print("LSTM 情感分类模型评估")
    print("=" * 60)

    # 1. 加载 Word2Vec 模型
    print("\n1. 加载 Word2Vec 模型...")
    w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH)

    # 2. 加载测试数据
    print("\n2. 加载测试数据...")
    X_test, y_test = load_data_as_vectors(TEST_DATA_PATH, w2v_model)
    print(f"测试集形状: {X_test.shape}")

    # 3. 转换为张量
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    # 4. 加载模型
    print("\n3. 加载 LSTM 模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = create_model(
        input_dim=EMBEDDING_DIM,
        hidden_dim=LSTM_HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        num_layers=LSTM_NUM_LAYERS
    ).to(device)

    model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    model.eval()

    # 5. 预测
    print("\n4. 开始预测...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(X_test), 32):
            batch_x = X_test[i:i + 32].to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_test[i:i + 32].numpy())

    # 6. 计算指标
    print("\n5. 评估结果:")
    print("=" * 60)

    # 准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"测试集准确率: {accuracy * 100:.2f}%")

    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(
        all_labels, all_preds,
        target_names=['负面', '中性', '正面']
    ))

    # 7. 绘制混淆矩阵
    print("\n6. 绘制混淆矩阵...")
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['负面', '中性', '正面']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # 添加数值标签
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('../models/confusion_matrix.png', dpi=150)
    plt.show()

    print("✅ 混淆矩阵已保存到: models/confusion_matrix.png")
    print("=" * 60)


if __name__ == '__main__':
    evaluate()