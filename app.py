# app.py
from flask import Flask, request, render_template, jsonify
import torch
import jieba
import pandas as pd
import io
from model import BiLSTM_Attention
from vocab import load_vocab

app = Flask(__name__)

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 加载模型和词汇表
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = load_vocab('data/vocab.pkl')
model = BiLSTM_Attention(len(vocab), 300, 128, 2).to(DEVICE)
model.load_state_dict(torch.load('models/best_model_emb.pth', map_location=DEVICE))
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
    sentiment = "正面" if pred == 1 else "负面"
    confidence = prob[0][pred].item()
    return sentiment, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': '请输入文本'})
    sentiment, confidence = predict(text)
    return jsonify({'sentiment': sentiment, 'confidence': confidence})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'})
    if not allowed_file(file.filename):
        return jsonify({'error': '只支持 CSV 文件格式'})

    # 读取 CSV 文件（假设包含 'text' 列）
    try:
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            return jsonify({'error': 'CSV 文件必须包含 "text" 列'})
    except Exception as e:
        return jsonify({'error': f'读取文件失败: {str(e)}'})

    # 批量预测
    results = []
    for idx, row in df.iterrows():
        text = str(row['text'])
        sentiment, confidence = predict(text)
        results.append({
            'index': idx,
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence
        })

    # 生成结果 CSV 供下载（可选）
    result_df = pd.DataFrame(results)
    output = io.StringIO()
    result_df.to_csv(output, index=False)
    output.seek(0)

    # 返回 JSON 数据（用于前端实时显示）
    return jsonify({
        'success': True,
        'total': len(results),
        'results': results,
        'csv_data': output.getvalue()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
