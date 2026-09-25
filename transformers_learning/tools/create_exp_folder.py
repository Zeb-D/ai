import os

import config


def create_exp_folder():
    """
    创建实验文件夹结构：run/train/exp（或 exp1, exp2...）
    如果 exp 文件夹不存在或为空，则直接使用它；否则依次尝试 exp1, exp2...
    返回 (exp_folder, weights_folder)
    """
    # 基础目录：data/train
    base_dir = config.model_dir
    os.makedirs(base_dir, exist_ok=True)

    def is_empty(path):
        """判断文件夹是否不存在或为空（无任何文件/子目录）"""
        if not os.path.exists(path):
            return True
        if os.path.isdir(path):
            return len(os.listdir(path)) == 0
        return False  # 如果是文件，视为非空

    exp_num = 0
    while True:
        # 第一个尝试 exp，之后 exp1, exp2...
        exp_name = "exp" if exp_num == 0 else f"exp{exp_num}"
        exp_folder = os.path.join(base_dir, exp_name)

        if is_empty(exp_folder):
            # 创建 exp 文件夹（如果不存在）
            os.makedirs(exp_folder, exist_ok=True)
            # 创建 weights 子文件夹
            weights_folder = os.path.join(exp_folder, "weights")
            os.makedirs(weights_folder, exist_ok=True)
            return exp_folder, weights_folder

        exp_num += 1
