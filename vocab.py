import pickle
import pandas as pd
from collections import Counter

def build_vocab(train_path, max_size=50000):
    df = pd.read_csv(train_path)
    words = []
    for text in df['cut']:
        words.extend(text.split())
    counter = Counter(words)
    # 保留最常见的 max_size-2 个词，留出 PAD 和 UNK
    most_common = counter.most_common(max_size - 2)
    vocab = {word: idx+2 for idx, (word, _) in enumerate(most_common)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def save_vocab(vocab, path='data/vocab.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(path='data/vocab.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    vocab = build_vocab('data/train.csv')
    save_vocab(vocab)
    print(f"词汇表大小: {len(vocab)}")