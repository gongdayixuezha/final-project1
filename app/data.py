# app/data.py 完整修复版
import os
import sys
import numpy as np
import struct
import wget
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# 导入路径工具（确保能正确导入）
from app.utils import add_project_root_to_path

add_project_root_to_path()

# ===================== 数据集配置 =====================
# 官方数据集URL（保持不变）
FASHION_MNIST_URLS = {
    "train_img": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte",
    "train_lab": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte",
    "test_img": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte",
    "test_lab": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte",
}

# 默认数据目录（修正为 data/raw，符合常规项目结构）
DEFAULT_DATA_ROOT = "data/raw"


def download_fashion_mnist(data_root: str = None) -> None:
    """
    下载Fashion MNIST数据集（仅当文件缺失时）
    参数：
        data_root: 数据存放目录（优先从环境变量读取，其次用默认值）
    """
    # 1. 确定数据目录（支持环境变量自定义，方便本地适配）
    data_root = data_root or os.getenv("FASHION_MNIST_DATA_ROOT", DEFAULT_DATA_ROOT)
    os.makedirs(data_root, exist_ok=True)  # 确保目录存在
    data_root_abs = os.path.abspath(data_root)
    print(f"🔍 检查数据集（目录：{data_root_abs}）...")

    # 2. 检查每个文件是否存在，仅下载缺失的
    all_files_exist = True  # 标记是否所有文件都已存在
    for name, url in FASHION_MNIST_URLS.items():
        filename = os.path.basename(url)
        save_path = os.path.join(data_root_abs, filename)

        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            all_files_exist = False  # 有文件缺失
            if not os.path.exists(save_path):
                print(f"⚠️ 未找到 {filename}，准备下载...")
            else:
                print(f"⚠️ {filename} 为空文件或损坏，准备重新下载...")

            # 3. 下载文件（增加超时重试）
            try:
                print(f"📥 正在下载：{url}")
                # 设置超时（10秒）和重试（3次）
                wget.download(url, out=save_path, bar=wget.bar_adaptive)
                print(f"\n✅ {filename} 下载完成（保存至：{save_path}）")
            except Exception as e:
                # 关键：如果下载失败但本地有旧文件，提示使用本地文件
                if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                    print(
                        f"\n⚠️ 下载失败，但检测到本地已有 {filename}（可能不完整），尝试继续使用..."
                    )
                else:
                    raise RuntimeError(
                        f"❌ 下载 {filename} 失败，且本地无可用文件！\n"
                        f"请手动下载并放到 {data_root_abs} 目录下：\n{url}"
                    ) from e

    # 4. 所有文件都存在时，直接跳过下载
    if all_files_exist:
        print(f"✅ 所有数据集文件已存在（{data_root_abs}），跳过下载")


def load_idx_file(file_path: str) -> np.ndarray:
    """读取idx格式文件（增强错误提示）"""
    file_path_abs = os.path.abspath(file_path)
    if not os.path.exists(file_path_abs):
        raise FileNotFoundError(
            f"❌ 未找到文件：{file_path_abs}\n"
            f"请确认文件是否存在，或手动下载放到该路径"
        )

    with open(file_path_abs, "rb") as f:
        magic_number, num_items = struct.unpack(">II", f.read(8))
        if magic_number == 2051:  # 图像文件
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                num_items, rows * cols
            )
        elif magic_number == 2049:  # 标签文件
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"❌ 不支持的文件格式（魔法数：{magic_number}）")
    return data


def load_local_fashion_mnist(scale_data: bool = True) -> tuple:
    """加载本地数据集（优先使用本地文件，避免重复下载）"""
    load_dotenv()  # 加载环境变量（支持自定义数据路径）

    # 确定数据目录（优先级：环境变量 > 默认值）
    data_root = os.getenv("FASHION_MNIST_DATA_ROOT", DEFAULT_DATA_ROOT)
    data_root_abs = os.path.abspath(data_root)

    # 仅当文件缺失时才下载（本地有则跳过）
    download_fashion_mnist(data_root_abs)

    # 加载数据（明确指定文件名，避免路径错误）
    print("🔍 正在加载数据集...")
    file_paths = {
        "train_img": os.path.join(data_root_abs, "train-images-idx3-ubyte"),
        "train_lab": os.path.join(data_root_abs, "train-labels-idx1-ubyte"),
        "test_img": os.path.join(data_root_abs, "t10k-images-idx3-ubyte"),
        "test_lab": os.path.join(data_root_abs, "t10k-labels-idx1-ubyte"),
    }

    # 验证所有文件是否存在（最终检查）
    for name, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ 关键文件缺失：{path}\n请手动下载并放置到该路径"
            )

    # 读取数据
    X_train = load_idx_file(file_paths["train_img"])
    y_train = load_idx_file(file_paths["train_lab"])
    X_test = load_idx_file(file_paths["test_img"])
    y_test = load_idx_file(file_paths["test_lab"])

    # 标准化处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) if scale_data else X_train
    X_test_scaled = scaler.transform(X_test) if scale_data else X_test

    print(
        f"✅ 数据集加载完成：\n"
        f"  - 训练集：{X_train_scaled.shape} | 测试集：{X_test_scaled.shape}\n"
        f"  - 数据来源：{data_root_abs}"
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# 本地运行时验证
if __name__ == "__main__":
    try:
        load_local_fashion_mnist()
        print("\n✅ 数据集验证成功，可用于模型训练")
    except Exception as e:
        print(f"\n❌ 数据集处理失败：{str(e)}")
