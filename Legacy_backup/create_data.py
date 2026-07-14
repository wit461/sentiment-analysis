# create_data.py
import pandas as pd
import os

# 确保 data 目录存在
os.makedirs('data', exist_ok=True)

# 创建数据
data = {
    'text': [
        '这个产品非常好用，强烈推荐！',
        '质量太差了，用了三天就坏了',
        '价格一般，东西还行吧',
        '快递速度很快，包装完好',
        '客服态度很差，再也不来了',
        '性价比还可以，对得起这个价格',
        '真的很棒，已经回购好几次了',
        '垃圾产品，千万别买',
        '凑合用吧，没有想象中好',
        '功能齐全，操作简单，好评'
    ],
    'label': ['正面', '负面', '中性', '正面', '负面', '中性', '正面', '负面', '中性', '正面']
}

df = pd.DataFrame(data)

# 保存到 data/comments.csv
file_path = 'data/comments.csv'
df.to_csv(file_path, index=False, encoding='utf-8-sig')

print(f"✅ 数据文件已创建: {file_path}")
print(f"📊 共 {len(df)} 条数据")
print("\n数据预览:")
print(df)