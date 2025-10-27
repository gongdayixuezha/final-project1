# app/utils.py 公共工具函数（解决路径配置重复问题）
import os
import sys


def add_project_root_to_path() -> str:
    """
    计算并将项目根目录添加到Python模块搜索路径
    返回：项目根目录路径
    """
    current_script_path = os.path.abspath(__file__)
    app_dir = os.path.dirname(current_script_path)
    project_root = os.path.dirname(app_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ 根目录已加入模块搜索路径：{project_root}")
    else:
        print(f"✅ 根目录已在搜索路径中：{project_root}")

    return project_root
