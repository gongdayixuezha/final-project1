# app/model.py 完整修复版（解决ModuleNotFoundError+全功能优化）
# ===================== 第一步：优先添加项目根目录到Python搜索路径 =====================
import os
import sys

# 计算项目根目录（从当前脚本路径向上两级：model.py → app/ → 项目根目录）
current_script_path = os.path.abspath(
    __file__
)  # E:\VSproject\final-project\app\model.py
app_dir = os.path.dirname(current_script_path)  # E:\VSproject\final-project\app
project_root = os.path.dirname(app_dir)  # E:\VSproject\final-project

# 强制添加项目根目录到搜索路径（确保Python能找到'app'模块）
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✅ 已将项目根目录添加到sys.path：{project_root}")

# ===================== 第二步：导入内部模块和依赖库 =====================
from app.utils import add_project_root_to_path  # 现在可正常导入

# 冗余路径保障（双重确认）
project_root = add_project_root_to_path()

import argparse
from dotenv import load_dotenv
import mlflow
import joblib
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# 导入数据加载函数
from app.data import load_local_fashion_mnist

# ===================== 第三步：配置MLflow与常量 =====================
load_dotenv()
mlflow.set_experiment("Fashion-MNIST-Logistic-Regression-Experiment")
print(f"✅ MLflow实验已配置：Fashion-MNIST-Logistic-Regression-Experiment")

# 模型注册名称（统一管理）
MODEL_REGISTER_NAME = "Fashion-MNIST-Logistic-Regression-Model"


# ===================== 第四步：模型训练函数 =====================
def train_model(
    learning_rate: float = 0.1, max_iter: int = 1000
) -> Tuple[OneVsRestClassifier, float]:
    """训练Fashion MNIST逻辑回归模型（解决警告+注册验证）"""
    # 加载标准化数据
    X_train, X_test, y_train, y_test, scaler = load_local_fashion_mnist(scale_data=True)

    # 启动MLflow Run
    run_name = f"LR-lr{learning_rate}-iter{max_iter}-saga"
    with mlflow.start_run(run_name=run_name) as run:
        # 记录参数
        mlflow.log_param("model_type", "OneVsRestClassifier(LogisticRegression)")
        mlflow.log_param("base_solver", "saga")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("regularization_strength_C", round(1 / learning_rate, 4))
        mlflow.log_param("multi_class_strategy", "OneVsRest")
        mlflow.log_param("dataset", "Fashion MNIST (60000 train / 10000 test)")

        # 初始化模型（解决multi_class弃用警告）
        base_model = LogisticRegression(
            C=1 / learning_rate,
            solver="saga",
            max_iter=max_iter,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        model = OneVsRestClassifier(base_model)

        # 训练模型
        print(f"📌 开始训练模型：{run_name}")
        model.fit(X_train, y_train)

        # 评估性能
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(
            f"✅ 训练完成：\n"
            f"  - 训练准确率：{train_accuracy:.4f}\n"
            f"  - 测试准确率：{test_accuracy:.4f}\n"
            f"  - 实际迭代次数：{model.estimators_[0].n_iter_[0]}"
        )

        # 记录指标
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("final_iterations_used", model.estimators_[0].n_iter_[0])

        # 保存标准化器
        scaler_path = "fashion_mnist_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
        os.remove(scaler_path)
        print(f"✅ 标准化器已保存到MLflow：preprocessing/{scaler_path}")

        # 记录并注册模型
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="fashion-mnist-lr-model",
            signature=signature,
            registered_model_name=MODEL_REGISTER_NAME,
        )

        # 验证模型注册
        client = MlflowClient()
        try:
            client.get_registered_model(MODEL_REGISTER_NAME)
            latest_version = client.get_latest_versions(MODEL_REGISTER_NAME)[0]
            print(
                f"✅ 模型注册验证成功：\n"
                f"  - 模型名称：{MODEL_REGISTER_NAME}\n"
                f"  - 最新版本：v{latest_version.version}"
            )
        except mlflow.exceptions.MlflowException as e:
            raise RuntimeError(f"❌ 模型注册失败：{str(e)}") from e

    return model, test_accuracy


# ===================== 第五步：命令行参数解析 =====================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fashion MNIST模型训练脚本")
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="学习率（默认0.1）"
    )
    parser.add_argument(
        "--max_iter", type=int, default=1000, help="迭代次数（默认1000）"
    )
    return parser.parse_args()


# ===================== 第六步：执行入口 =====================
if __name__ == "__main__":
    args = parse_args()
    print("\n=== 开始Fashion MNIST模型训练 ===")
    print(f"📋 训练参数：learning_rate={args.learning_rate}, max_iter={args.max_iter}")

    try:
        model, test_acc = train_model(
            learning_rate=args.learning_rate, max_iter=args.max_iter
        )
        print(f"\n=== 训练结果汇总 ===")
        print(f"📊 测试准确率：{test_acc:.4f}（正常范围：0.88-0.91）")
        print(f"💡 查看MLflow：mlflow ui → http://localhost:5000")

        # 验证准确率合理性
        assert 0.80 < test_acc < 0.95, "❌ 准确率异常，可能数据或训练有误！"

    except Exception as e:
        print(f"\n❌ 训练失败：{str(e)}")
        sys.exit(1)
