import pandas as pd
import jieba
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/comments.csv')
# 确保列名为 'text' 和 'label'
df.columns = ['text', 'label']
# 删除缺失值
df = df.dropna()

# 确保标签是整数（0 负面，1 正面）
df['label'] = df['label'].astype(int)

# 检查数据量
print(f"数据总量: {len(df)}")
print(df['label'].value_counts())  # 应该输出 0 和 1 的数量

# 中文分词
print("正在分词...")
df['cut'] = df['text'].apply(lambda x: ' '.join(jieba.lcut(str(x))))

# 划分数据集 (60% train, 20% val, 20% test)
train, temp = train_test_split(df, test_size=0.4, random_state=42, stratify=df['label'])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

# 保存
train.to_csv('data/train.csv', index=False, encoding='utf-8')
val.to_csv('data/val.csv', index=False, encoding='utf-8')
test.to_csv('data/test.csv', index=False, encoding='utf-8')

print(f"训练集: {len(train)} 条")
print(f"验证集: {len(val)} 条")
print(f"测试集: {len(test)} 条")