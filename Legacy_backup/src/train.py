# -*- coding: utf-8 -*-
"""
模型训练脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_model, count_parameters
from embedding import load_data_as_vectors
from config import (
    TRAIN_DATA_PATH, VAL_DATA_PATH, LSTM_MODEL_PATH,
    EMBEDDING_DIM, LSTM_HIDDEN_SIZE, NUM_CLASSES,
    LSTM_NUM_LAYERS, DROPOUT, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    WORD2VEC_MODEL_PATH
)
from gensim.models import Word2Vec


def train():
    """训练主函数"""
    print("=" * 60)
    print("LSTM 情感分类模型训练")
    print("=" * 60)

    # 1. 加载 Word2Vec 模型
    print("\n1. 加载 Word2Vec 模型...")
    if not os.path.exists(WORD2VEC_MODEL_PATH):
        print(f"❌ Word2Vec 模型不存在: {WORD2VEC_MODEL_PATH}")
        print("请先运行 embedding.py 训练 Word2Vec")
        return

    w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH)
    print(f"✅ Word2Vec 加载成功，词汇表大小: {len(w2v_model.wv.key_to_index)}")

    # 2. 加载训练数据和验证数据
    print("\n2. 加载数据...")
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"❌ 训练数据不存在: {TRAIN_DATA_PATH}")
        print("请先运行 preprocess.py 预处理数据")
        return

    X_train, y_train = load_data_as_vectors(TRAIN_DATA_PATH, w2v_model)
    X_val, y_val = load_data_as_vectors(VAL_DATA_PATH, w2v_model)

    print(f"训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")

    # 3. 转换为 PyTorch 张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    # 4. 创建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 5. 创建模型
    print("\n3. 创建 LSTM 模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = create_model(
        input_dim=EMBEDDING_DIM,
        hidden_dim=LSTM_HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        num_layers=LSTM_NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    print(f"模型参数量: {count_parameters(model):,}")

    # 6. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. 训练循环
    print("\n4. 开始训练...")
    print("=" * 60)

    best_val_acc = 0

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_acc = 100 * train_correct / train_total

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = 100 * val_correct / val_total

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), LSTM_MODEL_PATH)
            print(f"✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")

        # 打印进度
        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
              f"Loss: {total_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")

    # 8. 训练完成
    print("\n" + "=" * 60)
    print(f"🎉 训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型已保存到: {LSTM_MODEL_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    train()