# d2l 基础知识的 张量算法
if __name__ == "__main__":
    import torch

    A = torch.arange(20, dtype=torch.float32).reshape(5, -1)
    B = torch.arange(20, dtype=torch.float32).reshape(-1, 4)
    print(f"A {A} \nB {B} \nC {A * B}")

    X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    a = 2
    print(f"X {a + X} \na {(a * X).shape} ")
    print(f"{X.sum(dim=0)}\n {X.sum(dim=1)}\n {X.sum(dim=2)}\n {X.sum(dim=2, keepdim=True)}")

    print(f"{X.mean(dim=0)}\n {X.mean(dim=1)}\n {X.mean(dim=2)}\n {X.mean(dim=2, keepdim=True)}")

    ### 点积操作
    x = torch.arange(4, dtype=torch.float32)
    y = torch.ones(4, dtype=torch.float32)
    print(f"{x} \n {y} \n {torch.dot(x, y)}")

    A = torch.arange(20, dtype=torch.float32).reshape(5, -1)
    print(f"A {A} \nx {x} \n{A * x} \n{torch.mv(A, x)}")

    ### 矩阵乘法
    print(f"A {A} \nB {B}\nB.T {B.T} \nA @ B.T {A @ B.T}")
    print(f"A @ B.T \n{torch.matmul(A, B.T)}")

    ### 范数操作
    print(f"norm A {torch.norm(A, keepdim=True)}")
