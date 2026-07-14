import torch
from torch.utils.data import Dataset
import pandas as pd


class SentimentDataset(Dataset):
    def __init__(self, csv_path, vocab, max_len=100):
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['cut']
        label = self.df.iloc[idx]['label']
        # 将分词后的文本转换为索引序列
        tokens = text.split()[:self.max_len]
        ids = [self.vocab.get(token, 1) for token in tokens]  # 1 = <UNK>
        # Padding
        ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

