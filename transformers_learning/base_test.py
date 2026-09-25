if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    # Q: [batch=2, heads=4, seq_len=8, head_dim=16]
    Q = torch.randn(2, 2, 2, 2)
    print(Q.shape)
    K = torch.randn(2, 2, 2, 2)
    print(K)

    # K.transpose(-2, -1): [2, 4, 16, 8]
    # 交换 seq_len 和 head_dim 两个维度 ， 即 每个小矩阵内 进行转置
    K_t = K.transpose(-2, -1)
    print("K_t: ", K_t)

    # matmul: [2, 4, 8, 16] @ [2, 4, 16, 8] → [2, 4, 8, 8]
    scores = torch.matmul(Q, K_t)

    print(scores.shape)  # torch.Size([2, 4, 8, 8])
    print(scores.size())

    position = torch.arange(12).unsqueeze(dim=1)  # 变成对应方向的矩阵
    print(f"position: {position}")

    a = torch.arange(0, 10, 2)
    print(a)

    pe = torch.zeros(3, 10)
    print(pe)


    def temperature_softmax(logits, temperature=1.0):
        # 先除以温度，再用 torch.softmax
        return torch.softmax(logits / temperature, dim=-1)


    # 示例
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    T = 2.0  # 高温，更平滑
    probs = temperature_softmax(logits, T)
    print(probs)
    T = 0.3  # 低温，更理智
    probs = temperature_softmax(logits, T)
    print(probs)

    import torch
    import torch.nn as nn

    print("定义一个线性层")
    linear = nn.Linear(3, 5) # 线性网络，即 W*X + b
    print(linear.weight)  # 形状 (8, 4)，随机小数
    print(linear.bias)  # 形状 (8,)，随机小数

    # 使用 Xavier 均匀分布初始化权重
    nn.init.xavier_uniform_(linear.weight)

    # 查看初始化后的权重范围
    print(linear.weight.min(), linear.weight.max())
