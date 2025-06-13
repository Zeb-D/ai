# text_generation_optimized.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging


# 检查环境
def check_environment():
    # 检查PyTorch版本
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version < (2, 1):
        print(f"警告: 检测到PyTorch版本 {torch.__version__}，建议升级到2.1.0或更高版本")
        print("使用PyTorch 1.12.0时可能需要手动指定模型参数类型")

    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU进行推理，速度较慢")

    # 检查Transformers版本
    import transformers
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Transformers版本: {transformers.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")


check_environment()

# 禁用不必要的警告
logging.set_verbosity_error()

# 模型ID
model_id = "gpt2"

# 加载分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"分词器加载成功: {model_id}")
except Exception as e:
    print(f"分词器加载失败: {e}")
    exit(1)

# 手动指定模型参数类型以兼容旧版PyTorch
try:
    # 尝试半精度加载（需GPU支持）
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # 半精度
            low_cpu_mem_usage=True
        )
    else:
        # CPU环境使用float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # 全精度
            low_cpu_mem_usage=True
        )
    print(f"模型加载成功: {model_id}")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("尝试使用CPU和全精度重新加载...")
    try:
        # 备选方案：强制CPU和全精度
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        print(f"模型已使用CPU加载: {model_id}")
    except Exception as e2:
        print(f"最终加载失败: {e2}")
        print("请确保PyTorch正确安装或升级到最新版本")
        exit(1)


# 生成函数
def generate_text(prompt, max_length=100, temperature=0.7):
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")

    # 将输入移至模型设备
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # 解码输出
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 测试
if __name__ == "__main__":
    prompt = "Once upon a time"
    print("\n生成文本:")
    print(generate_text(prompt, max_length=100))