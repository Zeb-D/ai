if __name__ == "__main__":
    import torch

    # 定义模型和损失函数
    w = torch.tensor([1.0], requires_grad=True)  # 初始权重 w=1.0
    b = torch.tensor([0.0], requires_grad=True)  # 初始偏置 b=0.0
    X = torch.tensor([2.0])  # 输入特征 X=2.0
    y_true = torch.tensor([3.0])  # 真实标签 y_true=3.0

    # 前向计算
    y_pred = w * X + b  # y_pred = 1.0 * 2.0 + 0.0 = 2.0
    loss = (y_pred - y_true) ** 2  # loss = (2.0 - 3.0)^2 = 1.0
    print(f'y_pred: {y_pred}')
    print(f'loss: {loss}')

    # 反向传播计算梯度
    loss.backward()

    print("梯度 w:", w.grad)  # 输出: tensor([-4.])
    print("梯度 b:", b.grad)  # 输出: tensor([-2.])

    # 手动验证梯度
    d_loss_d_ypred = 2 * (y_pred - y_true)  # ∂loss/∂y_pred = 2*(2.0-3.0) = -2.0
    d_ypred_d_w = X  # ∂y_pred/∂w = X = 2.0
    d_ypred_d_b = 1.0  # ∂y_pred/∂b = 1.0

    print("手动计算 w梯度:", d_loss_d_ypred * d_ypred_d_w)  # tensor([-4.])
    print("手动计算 b梯度:", d_loss_d_ypred * d_ypred_d_b)  # tensor([-2.])

    print("矩阵的梯度")
    # 定义多元素参数（d=3个特征，n=4个样本）
    w = torch.tensor([[2.0], [-1.0], [3.0]], requires_grad=True)  # shape (3, 1)
    X = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0],
                      [10.0, 11.0, 12.0]])  # shape (4, 3)：4个样本、3个特征
    b = torch.tensor(1.0, requires_grad=True)  # 标量 1.0
    y_pred = X @ w + b  # 矩阵乘法 (@) + 广播加法
    print("y_pred:\n", y_pred)
    # y_pred shape: (4, 1)

    y_true = torch.tensor([[7.0], [13.0], [19.0], [25.0]])  # 特征： shape (4, 1)
    loss = torch.mean((y_pred - y_true) ** 2)  # MSE损失：均方差

    loss.backward()  # 自动计算梯度
    print("梯度 w:\n", w.grad)  # shape (3, 1)
    print("梯度 b:", b.grad)  # 标量

    n = X.shape[0]
    residual = y_pred - y_true  # shape (4, 1)

    # 手动计算梯度
    manual_w_grad = (2 / n) * X.T @ residual  # X^T shape (3,4), residual shape (4,1)
    manual_b_grad = (2 / n) * torch.sum(residual)
    print("手动计算 n:", n)
    print("手动计算 w梯度:\n", manual_w_grad)
    print("手动计算 b梯度:", manual_b_grad)
