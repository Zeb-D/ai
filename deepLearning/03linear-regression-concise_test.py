# 对应资料：https://zh.d2l.ai/chapter_linear-networks/linear-regression-concise.html
if __name__ == "__main__":
    import numpy as np
    import torch
    from torch.utils import data
    from d2l import torch as d2l

    # 1. 数据准备部分
    # 设置真实参数
    true_w = torch.tensor([2, -3.4], dtype=torch.float32)  # 明确指定数据类型
    true_b = 4.2

    # 生成合成数据
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)


    # 2. 数据加载器部分
    def load_array(data_arrays, batch_size, is_train=True):
        """构造一个PyTorch数据迭代器
        参数:
            data_arrays: 包含特征和标签的元组 (features, labels)
            batch_size: 批量大小
            is_train: 是否打乱数据 (默认True)
        返回:
            DataLoader对象
        """
        # 创建TensorDataset包装特征和标签
        dataset = data.TensorDataset(*data_arrays)
        # 创建DataLoader，设置是否打乱数据
        return data.DataLoader(dataset, batch_size, shuffle=is_train)


    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    # 3. 模型定义部分
    from torch import nn

    # 创建顺序模型，包含单个线性层 (输入维度2，输出维度1)
    net = nn.Sequential(nn.Linear(2, 1))

    # 初始化模型参数 (PyTorch默认已有初始化，这里显式设置更合理的初始值)
    nn.init.normal_(net[0].weight, mean=0, std=0.01)  # 权重初始化
    nn.init.constant_(net[0].bias, val=0)  # 偏置初始化

    # 4. 训练配置部分
    loss = nn.MSELoss()  # 使用均方误差损失
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 添加优化器定义

    num_epochs = 3

    # 5. 训练循环部分
    for epoch in range(num_epochs):
        for X, y in data_iter:
            # 前向传播计算损失
            l = loss(net(X), y)

            # 反向传播
            trainer.zero_grad()  # 梯度清零
            l.backward()  # 计算梯度
            trainer.step()  # 参数更新

        # 每个epoch结束后评估整个数据集
        with torch.no_grad():  # 不计算梯度，节省内存
            l = loss(net(features), labels)
            print(f'epoch {epoch + 1}, loss {l.item():.4f}')  # 格式化输出损失值

    # 6. 结果评估部分
    w = net[0].weight.data
    b = net[0].bias.data

    print('\n模型参数评估:')
    print(f'真实权重: {true_w.numpy()}, 估计权重: {w.reshape(true_w.shape).numpy()}')
    print(f'真实偏置: {true_b:.2f}, 估计偏置: {b.item():.4f}')
    print(f'权重估计误差: {(true_w - w.reshape(true_w.shape)).numpy()}')
    print(f'偏置估计误差: {true_b - b.item():.4f}')
