import torch


def main():
    # 检查是否有 MPS（Metal Performance Shaders）可用，用于 M 系列芯片 GPU 加速
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

if __name__ == "__main__":
    main()