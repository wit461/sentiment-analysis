import torch
from model import BiLSTM_Attention
from vocab import load_vocab
import jieba

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = load_vocab('data/vocab.pkl')
model = BiLSTM_Attention(len(vocab), 300, 128, 2).to(DEVICE)
model.load_state_dict(torch.load('models/best_model.pth', map_location=DEVICE))
model.eval()
MAX_LEN = 100

def predict(text):
    words = jieba.lcut(text)
    ids = [vocab.get(w, 1) for w in words][:MAX_LEN]
    ids = ids + [0] * (MAX_LEN - len(ids))
    input_tensor = torch.tensor(ids).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(input_tensor)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(out, dim=1).item()
    print(f"负面概率: {prob[0][0].item():.4f}, 正面概率: {prob[0][1].item():.4f}")
    return "正面" if pred == 1 else "负面"

text = "酒店的服务有点差"
print(predict(text))