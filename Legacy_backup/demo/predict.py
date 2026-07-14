# -*- coding: utf-8 -*-
"""
情感分析演示程序 - 支持单条/批量文本预测
"""

import torch
import jieba
import numpy as np
import sys
import os

# 添加 src 到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Legacy_backup.src.model import create_model
from Legacy_backup.src.config import (
    EMBEDDING_DIM, LSTM_HIDDEN_SIZE, NUM_CLASSES,
    LSTM_NUM_LAYERS, MAX_SEQ_LEN, IDX_TO_LABEL,
    WORD2VEC_MODEL_PATH, LSTM_MODEL_PATH
)
from gensim.models import Word2Vec


class SentimentPredictor:
    """情感分析预测器"""

    def __init__(self, model_path=None, w2v_path=None):
        """
        初始化预测器
        model_path: LSTM 模型路径
        w2v_path: Word2Vec 模型路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载 Word2Vec
        w2v_path = w2v_path or WORD2VEC_MODEL_PATH
        print(f"加载 Word2Vec 模型: {w2v_path}")
        self.w2v_model = Word2Vec.load(w2v_path)

        # 加载 LSTM 模型
        model_path = model_path or LSTM_MODEL_PATH
        print(f"加载 LSTM 模型: {model_path}")
        self.model = create_model(
            input_dim=EMBEDDING_DIM,
            hidden_dim=LSTM_HIDDEN_SIZE,
            num_classes=NUM_CLASSES,
            num_layers=LSTM_NUM_LAYERS
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        print(f"✅ 模型加载成功！使用设备: {self.device}")

    def text_to_vector(self, text):
        """将文本转换为词向量矩阵"""
        # 分词
        words = jieba.lcut(text)

        # 转换为词向量
        matrix = []
        for word in words:
            if word in self.w2v_model.wv:
                matrix.append(self.w2v_model.wv[word])
            else:
                matrix.append(np.zeros(EMBEDDING_DIM))

        # Padding 或截断
        if len(matrix) < MAX_SEQ_LEN:
            matrix += [np.zeros(EMBEDDING_DIM)] * (MAX_SEQ_LEN - len(matrix))
        else:
            matrix = matrix[:MAX_SEQ_LEN]

        return torch.FloatTensor(matrix).unsqueeze(0).to(self.device)

    def predict(self, text):
        """预测单条文本的情感"""
        vec = self.text_to_vector(text)
        with torch.no_grad():
            output = self.model(vec)
            _, pred = torch.max(output, 1)

        sentiment = IDX_TO_LABEL[pred.item()]
        # 计算置信度
        probs = torch.softmax(output, dim=1)
        confidence = probs[0][pred].item()

        return sentiment, confidence

    def batch_predict(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            sentiment, confidence = self.predict(text)
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': f"{confidence * 100:.1f}%"
            })
        return results

    def predict_with_details(self, text):
        """预测并返回详细信息（包括各类别概率）"""
        vec = self.text_to_vector(text)
        with torch.no_grad():
            output = self.model(vec)
            probs = torch.softmax(output, dim=1)

        result = {
            'text': text,
            'probabilities': {}
        }

        for i, label in IDX_TO_LABEL.items():
            result['probabilities'][label] = probs[0][i].item()

        _, pred = torch.max(output, 1)
        result['sentiment'] = IDX_TO_LABEL[pred.item()]
        result['confidence'] = probs[0][pred].item()

        return result


def main():
    """演示主函数"""
    print("=" * 60)
    print("LSTM 情感分类系统演示")
    print("=" * 60)

    # 初始化预测器
    predictor = SentimentPredictor()

    # 测试样例
    test_texts = [
        "这个课程太棒了，老师讲得特别好，受益匪浅！",
        "物流太慢了，等了一周才到，差评！",
        "价格一般，质量还行，凑合用吧",
        "强烈推荐！性价比超高，已经回购好几次了",
        "什么垃圾产品，再也不会买了",
        "客服态度很好，问题解决得很及时",
        "产品功能齐全，操作简单易懂，好评"
    ]

    print("\n" + "=" * 60)
    print("批量预测结果:")
    print("=" * 60)

    results = predictor.batch_predict(test_texts)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. 文本: {result['text']}")
        print(f"   情感: {result['sentiment']} (置信度: {result['confidence']})")

    # 交互模式
    print("\n" + "=" * 60)
    print("交互模式（输入 'exit' 退出）")
    print("=" * 60)

    while True:
        text = input("\n请输入评论文本: ").strip()
        if text.lower() == 'exit':
            print("再见！")
            break
        if not text:
            continue

        result = predictor.predict_with_details(text)
        print(f"\n预测结果: {result['sentiment']}")
        print(f"置信度: {result['confidence'] * 100:.1f}%")
        print("各类别概率:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob * 100:.1f}%")


if __name__ == '__main__':
    main()