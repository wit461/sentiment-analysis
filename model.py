import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim*2, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim*2) 因为双向
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return context

class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pretrained_embeddings=None):
        super(BiLSTM_Attention, self).__init__()
        if pretrained_embeddings is not None:
            # 使用预训练的词向量矩阵
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            # 注意：pretrained_embeddings 的形状应为 (vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.5)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        context = self.attention(lstm_out)
        out = self.fc(context)
        return out