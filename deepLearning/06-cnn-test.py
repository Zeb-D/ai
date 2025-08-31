import torch
from torch import nn
from d2l import torch as d2l


# 卷积运算
def corr2d(X, K):
    # 二维互相关运算，输出大小为 (Xh - Kh+1) * (Xh - Kh +1)
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def test_corr2d():
    X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    Y = torch.tensor([[0, 1], [2, 3]])
    K = corr2d(X, Y)
    print(K)
    # print(corr2d(X.t(), Y)) t为转置


if __name__ == "__main__":
    # test_corr2d()
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2))
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])  # 1：0到1的中间变化，-1：1到0的变化
    Y = corr2d(X, K)
    # print(corr2d(X.t(), K))
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2

    for i in range(50):
        Y_hat = conv2d(X)
        loss = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        loss.sum().backward()
        # 迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if i % 5 == 0:
            print(f"epoch: {i}, loss: {loss.sum():.3f}")

    print(conv2d.weight.data.reshape((1, 2)))
