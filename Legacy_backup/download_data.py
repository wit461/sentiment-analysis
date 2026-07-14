# download_data.py
import pandas as pd
import os


def create_larger_dataset():
    """创建一个更大的模拟数据集（200条）"""

    # 正面评论模板
    positive_templates = [
        "{}，非常满意", "{}，强烈推荐", "{}，物超所值", "{}，质量很好",
        "{}，服务周到", "{}，物流很快", "{}，还会回购", "{}，很实用",
        "{}，操作简单", "{}，性价比高", "{}，包装精美", "{}，效果很棒"
    ]
    positive_words = ["这个产品", "商品", "东西", "服务", "物流", "质量", "价格", "设计"]

    # 负面评论模板
    negative_templates = [
        "{}，太差了", "{}，很失望", "{}，质量不好", "{}，不会再买",
        "{}，客服态度差", "{}，物流太慢", "{}，不值这个价", "{}，做工粗糙",
        "{}，有瑕疵", "{}，效果不好", "{}，描述不符", "{}，体验很差"
    ]
    negative_words = ["这个产品", "商品", "东西", "服务", "物流", "质量", "价格", "设计"]

    # 中性评论模板
    neutral_templates = [
        "{}，一般般", "{}，还行吧", "{}，凑合用", "{}，中规中矩",
        "{}，没什么特别", "{}，还可以", "{}，能接受", "{}，普普通通"
    ]
    neutral_words = ["这个产品", "商品", "东西", "服务", "物流", "质量", "价格", "设计"]

    data = []

    # 生成正面评论（80条）
    for i in range(80):
        import random
        template = random.choice(positive_templates)
        word = random.choice(positive_words)
        text = template.format(word)
        data.append([text, "正面"])

    # 生成负面评论（80条）
    for i in range(80):
        import random
        template = random.choice(negative_templates)
        word = random.choice(negative_words)
        text = template.format(word)
        data.append([text, "负面"])

    # 生成中性评论（40条）
    for i in range(40):
        import random
        template = random.choice(neutral_templates)
        word = random.choice(neutral_words)
        text = template.format(word)
        data.append([text, "中性"])

    df = pd.DataFrame(data, columns=['text', 'label'])

    # 打乱顺序
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 保存
    df.to_csv('data/comments.csv', index=False, encoding='utf-8-sig')

    print(f"✅ 已创建 {len(df)} 条数据")
    print(f"   正面: {(df['label'] == '正面').sum()} 条")
    print(f"   负面: {(df['label'] == '负面').sum()} 条")
    print(f"   中性: {(df['label'] == '中性').sum()} 条")
    print("\n数据预览:")
    print(df.head(10))


if __name__ == '__main__':
    create_larger_dataset()