# -*- coding: utf-8 -*-
"""Flask Web 服务：单条预测 + CSV 批量预测，支持 BERT / BiLSTM 双模型切换。

通过环境变量 MODEL_TYPE 选择模型（默认 bert）：
  - bert   : BERT-base-chinese 微调（models/best_model_bert.pth，测试集准确率 91.50%）
  - bilstm : BiLSTM + Attention + 预训练词向量（models/best_model_emb.pth，86.80%）
"""

import io
import os

import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request

MODEL_TYPE = os.environ.get("MODEL_TYPE", "bert").lower()
MAX_LEN = 100
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
_predictor = None


def _build_predictor():
    if MODEL_TYPE == "bilstm":
        import jieba
        from model import BiLSTM_Attention
        from vocab import load_vocab

        vocab = load_vocab("data/vocab.pkl")
        model = BiLSTM_Attention(len(vocab), 300, 128, 2)
        model.load_state_dict(torch.load("models/best_model_emb.pth", map_location=DEVICE))
        model.eval()

        def encode(text):
            ids = [vocab.get(w, 1) for w in jieba.lcut(str(text))][:MAX_LEN]
            return ids + [0] * (MAX_LEN - len(ids))

        def predict_single(text):
            x = torch.tensor([encode(text)], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0]
            pred = int(torch.argmax(probs))
            return pred, float(probs[pred])

        def predict_batch(texts):
            x = torch.tensor([encode(t) for t in texts], dtype=torch.long).to(DEVICE)
            preds, confs = [], []
            with torch.no_grad():
                for i in range(0, x.shape[0], BATCH_SIZE):
                    probs = torch.softmax(model(x[i:i + BATCH_SIZE]), dim=1)
                    p = torch.argmax(probs, dim=1)
                    preds.extend(p.cpu().numpy().tolist())
                    confs.extend(probs.gather(1, p.unsqueeze(1)).squeeze(1).cpu().numpy().tolist())
            return preds, confs

        model_name = "BiLSTM + Attention + 预训练词向量"
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
        model.load_state_dict(torch.load("models/best_model_bert.pth", map_location=DEVICE))
        model.eval()

        def predict_single(text):
            enc = tokenizer(str(text), max_length=128, truncation=True, padding="max_length",
                            return_tensors="pt")
            with torch.no_grad():
                logits = model(input_ids=enc["input_ids"].to(DEVICE),
                               attention_mask=enc["attention_mask"].to(DEVICE)).logits
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(probs))
            return pred, float(probs[pred])

        def predict_batch(texts):
            preds, confs = [], []
            with torch.no_grad():
                for i in range(0, len(texts), BATCH_SIZE):
                    enc = tokenizer([str(t) for t in texts[i:i + BATCH_SIZE]], max_length=128,
                                    truncation=True, padding=True, return_tensors="pt")
                    logits = model(input_ids=enc["input_ids"].to(DEVICE),
                                   attention_mask=enc["attention_mask"].to(DEVICE)).logits
                    probs = torch.softmax(logits, dim=1)
                    p = torch.argmax(probs, dim=1)
                    preds.extend(p.cpu().numpy().tolist())
                    confs.extend(probs.gather(1, p.unsqueeze(1)).squeeze(1).cpu().numpy().tolist())
            return preds, confs

        model_name = "BERT-base-chinese 微调"

    return predict_single, predict_batch, model_name


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = _build_predictor()
    return _predictor


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model")
def model_info():
    get_predictor()
    return jsonify({"model": _predictor[2], "device": str(DEVICE)})


@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "请输入文本"}), 400
    predict_single, _, model_name = get_predictor()
    pred, conf = predict_single(text)
    return jsonify({"sentiment": "正面" if pred == 1 else "负面",
                    "confidence": round(conf, 4), "model": model_name})


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    if "file" not in request.files:
        return jsonify({"error": "未上传文件"}), 400
    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "只支持 CSV 文件格式"}), 400
    try:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            return jsonify({"error": 'CSV 文件必须包含 "text" 列'}), 400
    except Exception as exc:
        return jsonify({"error": f"读取文件失败: {exc}"}), 400

    texts = [str(v) for v in df["text"].tolist()]
    _, predict_batch_fn, model_name = get_predictor()
    preds, confs = predict_batch_fn(texts)
    results = [
        {"index": idx, "text": texts[idx],
         "sentiment": "正面" if p == 1 else "负面",
         "confidence": round(c, 4)}
        for idx, (p, c) in enumerate(zip(preds, confs))
    ]
    output = io.StringIO()
    pd.DataFrame(results).to_csv(output, index=False)
    return jsonify({"success": True, "total": len(results),
                    "results": results, "csv_data": output.getvalue(),
                    "model": model_name})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
