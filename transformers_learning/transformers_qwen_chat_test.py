import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 必须放在 import transformers 之前

from transformers import AutoTokenizer, AutoModelForCausalLM

from common.device import get_device

# 指定模型ID（来自 Hugging Face Hub）
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 设置设备，优先使用GPU
device = get_device()
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

if __name__ == "__main__":
    print(tokenizer)
    print(model)

    inputs = tokenizer(
        [
            "Okay, tell me your name please",
            "1+1="
        ],
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).to(device)

    print(inputs)

    print(tokenizer.batch_decode(inputs["input_ids"]))

    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=32, )
    print(outputs)
    print(tokenizer.batch_decode(outputs[0]))
    print(tokenizer.batch_decode(outputs[1]))