# 基础镜像：Python 3.10轻量版本（适配项目依赖）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（Git解决MLflow警告，gcc用于编译依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*  # 清理缓存减小镜像体积

# 复制依赖清单并安装Python库（清华源加速）
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 复制项目代码
COPY . .

# 环境变量配置（严格遵循Docker语法）
# 1. 抑制Git-Python警告
# 2. MLflow变量：运行时从.env文件加载值（此处用=声明空值）
ENV GIT_PYTHON_REFRESH=quiet \
    MLFLOW_TRACKING_URI= \
    MLFLOW_REGISTRY_URI=

# 容器启动命令（运行模型训练脚本）
CMD ["python", "app/model.py"]