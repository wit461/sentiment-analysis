# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
from model import BiLSTM_Attention
from vocab import load_vocab
from dataset import SentimentDataset

# ========== 超参数 ==========
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 40
LR = 0.001
PATIENCE = 5
MAX_LEN = 100
USE_PRETRAINED_EMB = False
PRETRAINED_EMB_PATH = 'data/cc.zh.300.vec'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 加载词汇表 ==========
vocab = load_vocab('data/vocab.pkl')
print(f"词汇表大小: {len(vocab)}")

# ========== 加载数据集 ==========
train_ds = SentimentDataset('data/train.csv', vocab, MAX_LEN)
val_ds = SentimentDataset('data/val.csv', vocab, MAX_LEN)
test_ds = SentimentDataset('data/test.csv', vocab, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

print(f"训练集大小: {len(train_ds)}")
print(f"验证集大小: {len(val_ds)}")
print(f"测试集大小: {len(test_ds)}")


# ========== 自定义加载预训练词向量的函数 ==========
def load_pretrained_embeddings(vocab, embedding_path, embedding_dim=300):
    print("加载预训练词向量...")
    word2vec = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        # 处理头信息（可能是第一行包含"词数 维度"）
        first_line = f.readline().strip()
        if first_line and len(first_line.split()) == 2 and first_line.split()[0].isdigit():
            # 是头信息，跳过
            pass
        else:
            # 第一行可能是词向量，尝试解析
            parts = first_line.split()
            if len(parts) == embedding_dim + 1:
                word2vec[parts[0]] = np.array(parts[1:], dtype=np.float32)
        # 逐行解析
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            parts = line.split()
            if len(parts) != embedding_dim + 1:
                # 如果长度不对，可能是文件损坏或格式不同，跳过
                continue
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                word2vec[word] = vec
            except ValueError:
                continue
    print(f"成功加载 {len(word2vec)} 个词向量")

    vocab_size = len(vocab)
    emb_mat = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    hit = 0
    for word, idx in vocab.items():
        # 尝试原始词和全小写
        if word in word2vec:
            emb_mat[idx] = word2vec[word]
            hit += 1
        elif word.lower() in word2vec:
            emb_mat[idx] = word2vec[word.lower()]
            hit += 1
    print(f"命中率: {hit}/{vocab_size} ({100 * hit / vocab_size:.2f}%)")
    return torch.tensor(emb_mat, dtype=torch.float)

# ========== 加载预训练词向量 ==========
pretrained_emb = None
if USE_PRETRAINED_EMB and os.path.exists(PRETRAINED_EMB_PATH):
    pretrained_emb = load_pretrained_embeddings(vocab, PRETRAINED_EMB_PATH, EMBEDDING_DIM)
else:
    print("未启用预训练词向量，使用随机初始化")

# ========== 模型、优化器、损失函数 ==========
model = BiLSTM_Attention(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, 2, pretrained_embeddings=pretrained_emb).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ========== 类别权重 ==========
train_labels = [label for _, label in train_ds]
unique, counts = np.unique(train_labels, return_counts=True)
class_counts = dict(zip(unique, counts))
print(f"训练集类别分布: {class_counts}")
n_samples = len(train_labels)
n_classes = len(unique)
weight_0 = n_samples / (n_classes * class_counts.get(0, 1))
weight_1 = n_samples / (n_classes * class_counts.get(1, 1))
class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
print(f"类别权重: 负面={class_weights[0]:.3f}, 正面={class_weights[1]:.3f}")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量: {total_params:,}")


# ========== 训练函数 ==========
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


# ========== 训练循环 ==========
best_val_acc = 0
patience_counter = 0
os.makedirs('models', exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion)
    scheduler.step(val_loss)
    print(
        f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_model.pth')
        patience_counter = 0
        print("  -> 保存最佳模型")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"早停触发，停止训练 (最佳验证准确率: {best_val_acc:.4f})")
            break

# ========== 测试 ==========
model.load_state_dict(torch.load('models/best_model.pth'))
_, test_acc = eval_epoch(model, test_loader, criterion)
print(f"\n测试集准确率: {test_acc:.4f}")

model.eval()
all_preds, all_labels = [], []
for x, y in test_loader:
    x = x.to(DEVICE)
    out = model(x)
    preds = torch.argmax(out, dim=1).cpu().numpy()
    all_preds.extend(preds)
    all_labels.extend(y.numpy())
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=['负面', '正面']))