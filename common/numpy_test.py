import numpy as np

# 资料来自
# 通用函数（ufuncs）列表：
# https://numpy.org/doc/stable/reference/ufuncs.html
#
# 数学函数：
# https://numpy.org/doc/stable/reference/routines.math.html
#
# 线性代数：
# https://numpy.org/doc/stable/reference/routines.linalg.html

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 构建数组
    np.array([1, 2, 3])
    print(np.ones(3, dtype=str))
    print(np.random.random(3))

    print("构建矩")
    print(np.array([[1, 2], [3, 4]]))
    print(np.zeros((3, 2), dtype=int))
    print("zeros_like:", np.zeros_like([1, 2], dtype=int))  # 构建与参数数组一样的矩阵，但元素是0
    print("full_like:", np.full_like([1, 2], fill_value=10, dtype=int))  # 构建与参数数组一样的矩阵，元素填充为fill_value
    print("eye:", np.eye(2, 3, k=1, dtype=int))  # 生成一个2*3单位矩阵(对角线是1)，对角线偏移量为1

    print("提取子集")
    data = np.array([[1, 2], [3, 4], [5, 6]])
    print(data)
    print(data[0, 1])  # 0行1列 # 输出 2
    # print(data[3, 2]) # 3行2列 # 直接报错
    print(data[1:4])  # 输出1到4行元素，越界不会报错
    print(data[1:3, ])
    print(data[0:2, 0])  # 输出0到2行的 0列元素
    print(data[data > 2])  # 大于2元素

    print("数组转置与reshape()")
    data = np.array([[1, 2], [3, 4], [5, 6]])
    print(data)
    print(data.T)  # 你可以理解行与列的下标互换
    print("reshape")
    data = np.array([1, 2, 3, 4, 5, 6])
    print(data.reshape(2, 3))  # 将一维数组变成2行3列数组
    # print(data.reshape(2, 2)) # 元素剩余会报错
    # print(data.reshape(3, 3)) # 元素不够会报错

    print("数组运算")
    print("矩阵可以进行加减乘除等数学运算:")
    data = np.array([[1, 2], [3, 4]])
    ones = np.ones((2, 2))  # 1行2列

    ad = data + ones  # 对应下标进行相加
    print(ad)
    print(data - ones)  # 相减
    print(data * data)
    print(data / data)
    print(data * 1.6)

    print("最小值/最大值/平均值等")
    print(data.max())
    print(data.max(axis=1))  # 按行分组计算最大值
    print(data.min())
    print(data.sum())
    print("所有元素的乘积:", data.prod())  # 1*2*3*4 = 24
    print("标准差（std）:", data.std())  # √(方差) ≈ 1.118
    print("方差（var）:", data.var())  # 方差 = 1.25
    print("矩阵平方:", np.square(data))  # 矩阵平方
    print("矩阵sin:", np.sin(data))  # 矩阵里面的元素求sin
    print("矩阵exp:", np.exp(data))  # 矩阵里面的元素求exp，即指数函数 e的x方
    print("矩阵去重:", np.unique(data))

    print("点积运算(dot)")
    data1 = np.array([1, 2, 3])
    data2 = np.array([[1, 10], [100, 1000], [10000, 100000]])
    print(data1.dot(data2))  # 输出线性代数的矩阵相乘，不是简单上面的逐元素相乘*

    print("高级运算")
    print("均方差公式")
    n = 6
    predictions = np.array([[1, 2, 3], [4, 5, 6]])
    labels = np.ones((2, 3))
    error = (1 / n) * np.sum(np.square(predictions - labels))  # np.square 为矩阵的平方
    print(error)
