# fix_and_train.py
"""
一键修复并重新训练
"""
import os
import sys
import pandas as pd
import numpy as np

print("=" * 60)
print("LSTM 情感分析 - 一键修复与训练")
print("=" * 60)

# 步骤1：生成更好的训练数据
print("\n📊 步骤1：生成训练数据...")

# 真实风格的评论数据
data = []

# 正面评论 (100条)
positive = [
    ("很好用，推荐购买", "正面"),
    ("质量不错，服务也很好", "正面"),
    ("性价比很高，值得入手", "正面"),
    ("物流很快，包装完好", "正面"),
    ("客服态度好，问题解决了", "正面"),
    ("产品很棒，超出预期", "正面"),
    ("做工精细，很满意", "正面"),
    ("操作简单，老人都会用", "正面"),
    ("外观漂亮，很喜欢", "正面"),
    ("效果很好，会回购", "正面"),
    ("价格实惠，质量好", "正面"),
    ("安装方便，配件齐全", "正面"),
    ("售后及时，很负责", "正面"),
    ("功能强大，实用性强", "正面"),
    ("材质很好，手感不错", "正面"),
]

# 负面评论 (100条)
negative = [
    ("质量太差，用了就坏", "负面"),
    ("客服态度恶劣，气死人", "负面"),
    ("物流太慢，等了一周", "负面"),
    ("和描述不符，骗人的", "负面"),
    ("做工粗糙，全是划痕", "负面"),
    ("价格虚高，不值这个价", "负面"),
    ("噪音很大，无法忍受", "负面"),
    ("售后服务差，没人管", "负面"),
    ("包装破损，东西坏了", "负面"),
    ("说明书看不懂，难用", "负面"),
    ("有异味，刺鼻难闻", "负面"),
    ("尺寸不对，退货麻烦", "负面"),
    ("耗电太快，续航差", "负面"),
    ("经常死机，没法用", "负面"),
    ("配件缺失，不完整", "负面"),
]

# 中性评论 (50条)
neutral = [
    ("一般般，没有惊喜", "中性"),
    ("还行吧，凑合用", "中性"),
    ("中规中矩，普通产品", "中性"),
    ("价格适中，质量一般", "中性"),
    ("能用，但没什么特别", "中性"),
    ("和预期差不多", "中性"),
    ("没什么好说的", "中性"),
    ("正常水平，不突出", "中性"),
    ("还可以，对得起价格", "中性"),
    ("普通商品，没亮点", "中性"),
]

# 扩展数据
import random
for text, label in positive:
    data.append([text, label])
    for _ in range(5):  # 每条生成5个变体
        variant = text.replace("很", random.choice(["非常", "特别", "超级", "十分"]))
        data.append([variant, label])

for text, label in negative:
    data.append([text, label])
    for _ in range(5):
        variant = text.replace("太", random.choice(["非常", "特别", "超级", "真的"]))
        data.append([variant, label])

for text, label in neutral:
    data.append([text, label])
    for _ in range(3):
        data.append([text + random.choice(["吧", "嗯", "哦"]), label])

df = pd.DataFrame(data, columns=['text', 'label'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存
df.to_csv('data/comments.csv', index=False, encoding='utf-8-sig')
print(f"✅ 生成了 {len(df)} 条数据")
print(f"   正面: {(df['label']=='正面').sum()} 条")
print(f"   负面: {(df['label']=='负面').sum()} 条")
print(f"   中性: {(df['label']=='中性').sum()} 条")

# 步骤2：更新配置文件
print("\n⚙️ 步骤2：更新模型配置...")

config_path = 'src/config.py'
with open(config_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修改参数
content = content.replace('LSTM_HIDDEN_SIZE = 64', 'LSTM_HIDDEN_SIZE = 16')
content = content.replace('LSTM_NUM_LAYERS = 2', 'LSTM_NUM_LAYERS = 1')
content = content.replace('EPOCHS = 20', 'EPOCHS = 80')
content = content.replace('BATCH_SIZE = 16', 'BATCH_SIZE = 8')
content = content.replace('LEARNING_RATE = 0.001', 'LEARNING_RATE = 0.0005')

with open(config_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 配置已更新")

# 步骤3：运行预处理
print("\n🔄 步骤3：运行数据预处理...")
os.system('python src/preprocess.py')

# 步骤4：训练Word2Vec
print("\n🔤 步骤4：训练Word2Vec...")
os.system('python src/embedding.py')

# 步骤5：训练LSTM
print("\n🧠 步骤5：训练LSTM模型...")
os.system('python src/train.py')

print("\n" + "=" * 60)
print("🎉 训练完成！现在可以运行 demo/predict.py 测试效果")
print("=" * 60)