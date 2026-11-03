# 🧠 中文情感分析系统

基于 BiLSTM + Attention 与 BERT 微调的中文酒店评论情感二分类系统，支持单条实时预测和 CSV 批量预测。

## ✨ 功能特点

- 🔍 **单条预测**：输入中文评论，实时返回情感（正面/负面）和置信度
- 📁 **批量预测**：上传 CSV 文件（含 `text` 列），一键分析并返回结果表格
- 📊 **置信度可视化**：进度条展示模型预测的把握程度
- 📝 **预测记录**：自动保存最近 10 条预测记录，支持清空
- 🚀 **Web 服务**：基于 Flask 提供友好界面

## 🏗️ 模型架构

- **词嵌入**：随机初始化 300 维（可替换为腾讯预训练词向量）
- **编码器**：双层 BiLSTM（双向长短期记忆网络）
- **注意力机制**：对 LSTM 输出加权，聚焦关键情感词
- **分类器**：全连接层 + Softmax，输出二分类（正面/负面）

## 📊 数据集与性能指标

数据集：ChnSentiCorp 酒店评论二分类，共 7765 条，按 60/20/20 分层划分
（训练 4659 / 验证 1553 / 测试 1553）。测试集指标（RTX 3050 Laptop）：

| 模型 | 准确率 | 负面 F1 | 正面 F1 |
|------|--------|---------|---------|
| BiLSTM + Attention（随机词向量，40 epochs） | 85.13% | 0.775 | 0.889 |
| BiLSTM + Attention + cc.zh.300 预训练词向量 | 86.80% | 0.797 | 0.902 |
| BERT-base-chinese 微调（6 epochs，max_len 128） | 91.50% | 0.865 | 0.938 |
| RoBERTa-wwm-ext 微调（warmup + 余弦调度 + 类别权重） | 91.05% | 0.864 | 0.933 |
| BERT + RoBERTa 软投票集成 | 91.76% | 0.870 | 0.940 |

> 修复记录：`model.py` 中 Attention 层维度在训练后被误改为 `hidden_dim * 2`，
> 导致已保存权重加载失败；已修复并复测基线。
>
> 结论：该数据集（酒店评论二分类）用基座模型微调的上限约 91-92%，
> RoBERTa 未能超过 BERT；进一步提升需数据增强、交叉验证集成或更大模型。

## ⚖️ 模型对比与选择

| 模型 | 参数量 | CPU 单条延迟 | GPU 单条延迟 | 测试集准确率 |
|------|--------|--------------|--------------|--------------|
| BiLSTM + Attention + 预训练词向量 | 7.69M | 3.0 ms | 1.8 ms | 86.80% |
| BERT-base-chinese 微调 | 102.27M | 97.7 ms | 15.6 ms | 91.50% |
| RoBERTa-wwm-ext 微调 | 102.27M | ~97.7 ms | ~15.6 ms | 91.05% |

延迟为单条预测实测（RTX 3050 Laptop；批量推理时每条约 0.77 ms / 10.9 ms）。
Web 服务默认加载 BERT（`MODEL_TYPE=bert`），追求极低延迟时可切换
`MODEL_TYPE=bilstm`。综合推理速度与部署成本，轻量架构（BiLSTM）适合实时与大批量场景。

## 🚀 运行

```powershell
# 默认 BERT 模型（需可访问 HuggingFace 或设置 HF_ENDPOINT 镜像）
python app.py

# 切换为轻量 BiLSTM 模型（CPU 3ms/条）
$env:MODEL_TYPE = "bilstm"
python app.py
```

## 📁 项目结构

├── data/ # 数据集、词汇表
├── models/ # 保存的训练模型
├── experiments/ # 对比实验脚本（基线评估、词向量、BERT）
├── templates/ # Flask 前端页面
├── app.py # Web 服务入口
│（默认加载 models/best_model_emb.pth，CPU 即可实时预测）
├── train.py # 模型训练脚本
├── preprocess.py # 数据预处理
├── model.py # BiLSTM+Attention 定义
├── dataset.py # PyTorch Dataset
├── vocab.py # 词汇表构建
├── evaluate.py # 模型评估
├── requirements.txt # 依赖列表
└── README.md # 项目说明
