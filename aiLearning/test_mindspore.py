import mindspore as ms
from mindspore import nn, Tensor
from mindspore.dataset import vision, transforms
import mindspore.dataset as ds
import numpy as np
import matplotlib.pyplot as plt


# 1. 数据加载与增强（兼容训练/验证/测试）
def create_mnist_dataset(data_path, batch_size=32, usage="train"):
    """创建MNIST数据集管道

    Args:
        data_path: 数据集根目录（包含MNIST二进制文件）
        batch_size: 批大小
        usage: 数据集类型 ("train"|"test")
    """
    # 检查参数合法性
    if usage not in ["train", "test"]:
        raise ValueError("usage must be 'train' or 'test'")

    # 加载原始数据集
    mnist_ds = ds.MnistDataset(data_path, usage=usage)

    # 定义图像变换
    resize_op = vision.Resize((32, 32), interpolation=vision.Inter.LINEAR)
    rescale_op = vision.Rescale(1.0 / 255.0, 0.0)  # 归一化到[0,1]
    hwc2chw_op = vision.HWC2CHW()  # 转为MindSpore需要的CHW格式

    # 训练集增强（仅对训练数据）
    if usage == "train":
        transform_ops = [
            vision.RandomRotation(degrees=15),
            vision.RandomCrop(28, padding=4),
            resize_op,
            rescale_op,
            hwc2chw_op
        ]
    else:
        transform_ops = [resize_op, rescale_op, hwc2chw_op]

    # 应用变换
    mnist_ds = mnist_ds.map(operations=transform_ops, input_columns="image")
    mnist_ds = mnist_ds.map(operations=transforms.TypeCast(ms.int32), input_columns="label")

    # 批处理与shuffle
    if usage == "train":
        mnist_ds = mnist_ds.shuffle(buffer_size=10000)

    return mnist_ds.batch(batch_size, drop_remainder=True)


# 2. 定义LeNet-5模型
class LeNet5(nn.Cell):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_classes)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# 3. 训练与验证
def train_and_validate():
    # 环境初始化
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    # 数据加载
    train_data = create_mnist_dataset("./datasets/MNIST_Data", batch_size=64, usage="train")
    val_data = create_mnist_dataset("./datasets/MNIST_Data", batch_size=64, usage="test")  # 用测试集作验证

    # 模型初始化
    model = LeNet5()

    # 损失函数与优化器
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(model.trainable_params(), learning_rate=0.01, momentum=0.9)

    # 定义训练步
    def train_step(data, label):
        def forward_fn(data, label):
            logits = model(data)
            loss = loss_fn(logits, label)
            return loss

        loss, grads = ms.value_and_grad(forward_fn, None, optimizer.parameters)(data, label)
        optimizer(grads)
        return loss

    # 训练循环
    epochs = 1
    for epoch in range(epochs):
        # 训练阶段
        model.set_train()
        train_losses = []
        for batch, (data, label) in enumerate(train_data.create_tuple_iterator()):
            loss = train_step(data, label)
            train_losses.append(loss.asnumpy())

            if batch % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch}], Loss: {loss.asnumpy():.4f}")

        # 验证阶段
        model.set_train(False)
        correct = 0
        total = 0
        for data, label in val_data.create_tuple_iterator():
            outputs = model(data)
            predicted = outputs.argmax(1)
            correct += (predicted == label).sum().asnumpy()
            total += label.shape[0]

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {np.mean(train_losses):.4f}, "
              f"Val Acc: {val_acc:.2f}%")

    # 保存训练好的模型权重
    ms.save_checkpoint(model, "lenet5.ckpt")
    print("模型权重已保存到 lenet5.ckpt")

# 4. 测试函数
def evaluate_testset():
    # 1. 加载测试数据
    test_data = create_mnist_dataset("./datasets/MNIST_Data", batch_size=64, usage="test")

    # 2. 初始化模型（结构必须与训练时一致）
    model = LeNet5()

    # 3. 加载预训练权重
    try:
        param_dict = ms.load_checkpoint("lenet5.ckpt")
        ms.load_param_into_net(model, param_dict)
        print("权重加载成功")
    except:
        raise FileNotFoundError("找不到权重文件 lenet5.ckpt，请先训练模型")

    # 4. 评估模式
    model.set_train(False)

    # 5. 计算准确率
    correct = 0
    total = 0
    for data, label in test_data.create_tuple_iterator():
        outputs = model(data)
        predicted = outputs.argmax(1)
        correct += (predicted == label).sum().asnumpy()
        total += label.shape[0]

    print(f"测试集准确率: {100 * correct / total:.2f}%")

# 5. 可视化样本
def visualize_samples():
    dataset = create_mnist_dataset("./datasets/MNIST_Data", batch_size=9, usage="train")
    for batch in dataset.create_dict_iterator():
        images = batch["image"]
        labels = batch["label"]

        plt.figure(figsize=(8, 8))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.title(f"Label: {labels[i].asnumpy()}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        break  # 只显示一个batch


if __name__ == "__main__":
    # 可视化数据样本（可选）
    visualize_samples()

    # 训练与验证 （首先/每次会生成lenet5.ckpt）
    # train_and_validate()

    # 测试（需先训练保存模型）
    evaluate_testset()