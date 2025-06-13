import math
import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


def synthetic_data(w, b, num_examples):
    """
    生成人工合成的线性回归数据集
    参数:
        w (torch.Tensor): 真实的权重向量
        b (float): 真实的偏置项
        num_examples (int): 样本数量
    返回:
        X (torch.Tensor): 特征矩阵 (num_examples × len(w))
        y (torch.Tensor): 标签向量 (num_examples × 1)
    """
    # 生成特征矩阵，从标准正态分布采样
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算无噪声的标签值
    y = torch.matmul(X, w) + b
    # 添加高斯噪声
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """
    小批量数据迭代器
    参数:
        batch_size (int): 批量大小
        features (torch.Tensor): 特征矩阵
        labels (torch.Tensor): 标签向量
    生成:
        (features_batch, labels_batch): 每个批次的特征和标签
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 随机打乱数据

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失函数"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降
    参数:
        params (list): 包含需要更新的参数 [w, b]
        lr (float): 学习率
        batch_size (int): 批量大小
    """
    with torch.no_grad():  # 不跟踪梯度计算
        for param in params:
            param -= lr * param.grad / batch_size  # 参数更新
            param.grad.zero_()  # 梯度清零


if __name__ == "__main__":
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    random.seed(42)

    # 真实参数
    true_w = torch.tensor([2, -3.4], dtype=torch.float32)
    true_b = 4.2

    # 生成数据集
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 可视化数据
    d2l.set_figsize()
    plt.scatter(features[:, 1].detach().numpy(),
                labels.detach().numpy(), 1)
    plt.xlabel("Feature 1")
    plt.ylabel("Label")
    plt.title("Synthetic Data Distribution")
    # plt.show()

    # 初始化模型参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    print(f'初始化w:{w}')
    print(f'初始化b:{b}')

    # 训练参数
    batch_size = 10
    lr = 0.03
    num_epochs = 3

    # 训练循环
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            # 计算预测值和损失
            y_hat = linreg(X, w, b)
            l = squared_loss(y_hat, y)

            # 反向传播计算梯度
            l.sum().backward()

            # 使用SGD更新参数
            sgd([w, b], lr, batch_size)

        # 每个epoch结束后评估整个数据集的损失
        with torch.no_grad():
            train_l = squared_loss(linreg(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    # 输出训练结果
    print(f'真实w: {true_w}, 估计w: {w.reshape(true_w.shape)}')
    print(f'真实b: {true_b}, 估计b: {b.item()}')
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b.item()}')

