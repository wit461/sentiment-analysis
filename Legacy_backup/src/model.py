# -*- coding: utf-8 -*-
"""
LSTM 模型定义
"""

import torch
import torch.nn as nn


class LSTMSentiment(nn.Module):
    """基于 LSTM 的情感分类模型"""

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(LSTMSentiment, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Dropout 层（防止过拟合）
        self.dropout = nn.Dropout(dropout)

        # 全连接层（输出层）
        # *2 因为双向 LSTM
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        前向传播
        x shape: (batch_size, seq_len, input_dim)
        """
        # 初始化隐藏状态和细胞状态
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)

        # LSTM 层
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        out = lstm_out[:, -1, :]

        # Dropout
        out = self.dropout(out)

        # 全连接层
        out = self.fc(out)

        return out


def create_model(input_dim=128, hidden_dim=64, num_classes=3, num_layers=2, dropout=0.5):
    """创建模型的工厂函数"""
    model = LSTMSentiment(input_dim, hidden_dim, num_classes, num_layers, dropout)
    return model


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试代码
if __name__ == '__main__':
    model = create_model()
    print(model)
    print(f"\n模型参数量: {count_parameters(model):,}")

    # 测试前向传播
    test_input = torch.randn(4, 50, 128)  # (batch, seq_len, input_dim)
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")