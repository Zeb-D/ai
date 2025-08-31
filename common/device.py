import os
import torch


def get_device():
    device = torch.device("cpu")
    # 优先使用 CUDA，其次使用 MPS，最后使用 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS-(Metal Performance Shaders) for GPU acceleration")
    # 检查 DirectML (Windows 上的 DirectX 12 加速)
    elif os.name == 'nt':  # Windows 系统
        try:
            import torch_directml  # 在windows 执行 pip install torch-directml

            dml_device = torch_directml.device()
            device = dml_device
            print(f"Using DirectML for GPU acceleration on device: {torch_directml.device_name(dml_device)}")
        except ImportError:
            print("DirectML not installed. Using CPU.")
            print("Install DirectML with: pip install torch-directml")
    return device


if __name__ == "__main__":
    print(get_device())
