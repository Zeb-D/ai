import math

import numpy as np


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


if __name__ == "__main__":
    from d2l import torch as d2l
    import matplotlib.pyplot as plt

    # 使用numpy进行可视化
    x = np.arange(-7, 7, 0.01)
    print(x)

    # 均值和标准差对
    params = [(0, 1), (0, 2), (3, 1)]
    # 绘制图形
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params],
             xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5),
             legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])

    # 显示图形（如果不在Jupyter中运行）
    plt.show()
