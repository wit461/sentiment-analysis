# -*- coding: utf-8 -*-
"""
数据预处理脚本 - 加载数据、分词、划分数据集
"""

import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from config import (
    RAW_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    LABEL_MAP, DATA_DIR
)
import os


def load_and_preprocess():
    """加载CSV并进行预处理"""
    print("=" * 50)
    print("开始数据预处理...")
    print("=" * 50)

    # 检查数据文件是否存在
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ 数据文件不存在: {RAW_DATA_PATH}")
        print("请先在 data/comments.csv 中创建示例数据")
        return None

    # 加载数据
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"原始数据量: {len(df)} 条")

    # 去除空值
    df = df.dropna()
    print(f"去除空值后: {len(df)} 条")

    # 去重
    df = df.drop_duplicates(subset=['text'])
    print(f"去重后: {len(df)} 条")

    # 标签映射
    df['label'] = df['label'].map(LABEL_MAP)

    # 去除无效标签
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # 中文分词
    print("正在进行中文分词...")
    df['cut_text'] = df['text'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 显示分词样例
    print("\n分词样例:")
    print(f"原文: {df['text'].iloc[0]}")
    print(f"分词: {df['cut_text'].iloc[0]}")

    return df


def split_dataset(df):
    """划分训练集、验证集、测试集 (6:2:2)"""
    print("\n" + "=" * 50)
    print("划分数据集...")
    print("=" * 50)

    # 先分训练集和临时集
    train, temp = train_test_split(df, test_size=0.4, random_state=42)
    # 再分验证集和测试集
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # 保存到文件
    train.to_csv(TRAIN_DATA_PATH, index=False, encoding='utf-8-sig')
    val.to_csv(VAL_DATA_PATH, index=False, encoding='utf-8-sig')
    test.to_csv(TEST_DATA_PATH, index=False, encoding='utf-8-sig')

    print(f"训练集: {len(train)} 条 -> {TRAIN_DATA_PATH}")
    print(f"验证集: {len(val)} 条 -> {VAL_DATA_PATH}")
    print(f"测试集: {len(test)} 条 -> {TEST_DATA_PATH}")

    # 显示标签分布
    print("\n标签分布:")
    print(
        f"训练集 - 负面:{(train['label'] == 0).sum()}, 中性:{(train['label'] == 1).sum()}, 正面:{(train['label'] == 2).sum()}")
    print(
        f"验证集 - 负面:{(val['label'] == 0).sum()}, 中性:{(val['label'] == 1).sum()}, 正面:{(val['label'] == 2).sum()}")
    print(
        f"测试集 - 负面:{(test['label'] == 0).sum()}, 中性:{(test['label'] == 1).sum()}, 正面:{(test['label'] == 2).sum()}")

    return train, val, test


def main():
    """主函数"""
    df = load_and_preprocess()
    if df is not None:
        split_dataset(df)
        print("\n✅ 数据预处理完成！")
    else:
        print("\n❌ 数据预处理失败，请检查数据文件")


if __name__ == '__main__':
    main()