# 使用官方 Python 3.9 轻量级镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（jieba 分词需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用清华源加速）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目所有文件
COPY . .

# 暴露 Flask 默认端口
EXPOSE 5000

# 启动命令
CMD ["python", "app.py"]