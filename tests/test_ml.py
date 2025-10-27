# tests/test_ml.py 最终修复版（解决 NameError + 保留原有功能）
# ===================== 第一步：配置路径 =====================
import os
import sys
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mlflow.pyfunc import PyFuncModel

# 配置项目根目录到搜索路径
current_test_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_test_path)
project_root = os.path.dirname(tests_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入公共路径工具
from app.utils import add_project_root_to_path

add_project_root_to_path()

# ===================== 第二步：导入项目模块（含常量） =====================
from app.model import load_local_fashion_mnist, train_model
from app.predict import (
    load_trained_model_and_scaler,
    predict_fashion_mnist,
    FASHION_MNIST_MODEL_NAME,  # 导入常量，解决 NameError
)


# ===================== 第三步：测试用例 =====================
def test_data_loading():
    """测试Fashion MNIST数据加载：验证维度、类别和标准化器"""
    X_train, X_test, y_train, y_test, scaler = load_local_fashion_mnist()
    # 维度验证
    assert X_train.shape == (60000, 784), f"训练集维度错误：{X_train.shape}"
    assert X_test.shape == (10000, 784), f"测试集维度错误：{X_test.shape}"
    assert y_train.shape == (60000,), f"训练标签维度错误：{y_train.shape}"
    assert y_test.shape == (10000,), f"测试标签维度错误：{y_test.shape}"
    # 类别验证
    assert set(y_train) == set(range(10)), f"训练集类别错误：{set(y_train)}"
    # 标准化器验证
    assert isinstance(scaler, StandardScaler), "标准化器类型错误"


def test_model_training():
    """测试模型训练：收敛性+准确率达标"""
    # 用1000次迭代确保收敛，准确率阈值0.83（留缓冲）
    model, accuracy = train_model(learning_rate=0.1, max_iter=1000)
    # 准确率验证
    assert accuracy >= 0.83, f"准确率过低（{accuracy:.4f}），需≥0.83"
    # 模型类型验证（逻辑回归原生支持多分类ovr）
    assert isinstance(model, LogisticRegression), "模型不是LogisticRegression类型"
    assert hasattr(model, "predict"), "模型缺少predict方法"


def test_model_prediction():
    """测试模型预测：类型兼容+结果正确"""
    # 使用导入的常量，避免 NameError
    model, scaler = load_trained_model_and_scaler(model_name=FASHION_MNIST_MODEL_NAME)
    # 模型类型验证
    assert isinstance(model, PyFuncModel), "模型不是MLflow PyFuncModel"
    assert isinstance(scaler, StandardScaler), "标准化器类型错误"
    # 生成测试样本
    test_sample = np.random.normal(loc=0, scale=1, size=784).tolist()
    # 预测
    class_index, class_name = predict_fashion_mnist(model, scaler, test_sample)
    # 结果验证
    assert isinstance(
        class_index, (int, np.integer)
    ), f"类别索引类型错误：{type(class_index)}，需int或numpy.int"
    assert 0 <= class_index <= 9, f"类别索引超出范围：{class_index}"
    assert class_name in [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ], f"未知类别名称：{class_name}"


def test_prediction_with_invalid_input():
    """测试异常输入：维度错误时正确报错"""
    # 使用导入的常量，避免 NameError
    model, scaler = load_trained_model_and_scaler(model_name=FASHION_MNIST_MODEL_NAME)
    # 测试100维无效样本（需784维）
    invalid_sample = [0.1] * 100
    with pytest.raises(ValueError) as excinfo:
        predict_fashion_mnist(model, scaler, invalid_sample)
    assert "输入特征维度错误！需784个数值" in str(excinfo.value)
