# -*- coding: utf-8 -*-
"""
配置文件 - 存放所有超参数和路径配置
"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 数据文件路径
RAW_DATA_PATH = os.path.join(DATA_DIR, 'comments.csv')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
VAL_DATA_PATH = os.path.join(DATA_DIR, 'val.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')

# Word2Vec 模型路径
WORD2VEC_MODEL_PATH = os.path.join(DATA_DIR, 'word2vec.model')

# LSTM 模型路径
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_sentiment.pth')

# 创建必要的目录
os.makedirs(MODEL_DIR, exist_ok=True)

# ========== 模型超参数 ==========
MAX_SEQ_LEN = 50          # 最大序列长度（每条文本最多保留50个词）
EMBEDDING_DIM = 128       # 词向量维度
LSTM_HIDDEN_SIZE = 16     # LSTM 隐藏层维度
LSTM_NUM_LAYERS = 1       # LSTM 层数
DROPOUT = 0.5             # Dropout 比例

# ========== 训练超参数 ==========
BATCH_SIZE = 8           # 批次大小（如果内存不够可以改小）
EPOCHS = 80               # 训练轮数
LEARNING_RATE = 0.0005     # 学习率

# ========== 标签映射 ==========
LABEL_MAP = {'负面': 0, '中性': 1, '正面': 2}
IDX_TO_LABEL = {0: '负面', 1: '中性', 2: '正面'}
NUM_CLASSES = 3

# ========== Word2Vec 参数 ==========
W2V_VECTOR_SIZE = 128
W2V_WINDOW = 5
W2V_MIN_COUNT = 1
W2V_WORKERS = 4

print("✅ 配置加载完成")