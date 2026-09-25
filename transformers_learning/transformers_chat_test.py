import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 必须放在 import transformers 之前

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from common.device import get_device

# 指定模型ID（来自 Hugging Face Hub）
# Helsinki-NLP/opus-mt-en-zh 是一个英文到中文的翻译模型
model_id = "Helsinki-NLP/opus-mt-en-zh"

# 设置设备，优先使用GPU
device = get_device()
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型，并将其移动到指定设备
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

print("模型和分词器加载完成！")


def translate(text: str) -> str:
    """将英文文本翻译成中文。"""
    # 编码输入文本
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    print(model_inputs.input_ids)

    # 使用模型生成翻译
    # max_new_tokens 控制了模型最多能生成多少个新的 Token
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    print(generated_ids)

    # 解码生成的 Token ID
    translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return translation


###############################################################################
# 10. Hugging Face 模型文件内部结构验证（对应《https://zhuanlan.zhihu.com/p/2085763781674657188》第四节）
###############################################################################
# 目的：复现文档第四节的实测结论，直观展示「下载到本地的那几个文件」在
#      transformers 框架内部究竟是如何被使用的：
#        config.json            -> 决定用哪个类 + 决定张量形状（造骨架）
#        pytorch_model.bin      -> state_dict，用来填充骨架参数
#        vocab.json / *.spm     -> 文本 <-> token id 的映射与切分
#        generation_config.json -> generate() 的默认策略
#
# 运行： python transformer_simple_test.py
# 依赖： pip install transformers sentencepiece（模型会自动从 HF 下载并缓存）


def hf_verify_config():
    """[4.1] config.json → 决定模型类与张量形状。"""
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import (
        MODEL_MAPPING_NAMES,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    )

    print("=" * 72)
    print("[4.1] config.json → 决定用哪个类、按什么形状造骨架")
    cfg = AutoConfig.from_pretrained(model_id)
    print("config 实际类           :", type(cfg).__name__)
    print("model_type              :", cfg.model_type)
    print("architectures           :", cfg.architectures)
    print("d_model                 :", cfg.d_model)
    print("encoder_layers/decoder  :", cfg.encoder_layers, "/", cfg.decoder_layers)
    print("encoder_ffn_dim         :", cfg.encoder_ffn_dim)
    print("vocab_size              :", cfg.vocab_size)
    print("max_position_embeddings :", cfg.max_position_embeddings)
    print("decoder_start_token_id  :", cfg.decoder_start_token_id)
    print("MODEL_MAPPING_NAMES['marian'] ->",
          MODEL_MAPPING_NAMES.get(cfg.model_type))
    print("MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING['marian'] ->",
          MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES.get(cfg.model_type))
    return cfg


def hf_verify_weights():
    """[4.2] pytorch_model.bin → 张量数量 / 参数量 / 关键形状。"""
    import torch
    from huggingface_hub import hf_hub_download

    print("=" * 72)
    print("[4.2] pytorch_model.bin → 用来填充 config 造出的骨架")

    try:
        weight_path = hf_hub_download(model_id, "pytorch_model.bin")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    except Exception:
        # 兼容仓库改用 safetensors 的情况
        from safetensors.torch import load_file
        weight_path = hf_hub_download(model_id, "model.safetensors")
        state_dict = load_file(weight_path)

    total = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
    keys = list(state_dict.keys())
    print("权重文件路径 :", weight_path)
    print("文件大小     :", os.path.getsize(weight_path), "bytes")
    print("张量数量     :", len(state_dict))
    print("参数量合计   :", f"{total:,}", f"(≈{total / 1e6:.1f}M)")
    print("含 encoder   :", sum(1 for k in keys if "encoder" in k), "个")
    print("含 decoder   :", sum(1 for k in keys if "decoder" in k), "个")
    print("前 6 个参数名与形状:")
    for k in keys[:6]:
        print("   ", k, tuple(state_dict[k].shape))
    print("后 4 个参数名与形状:")
    for k in keys[-4:]:
        print("   ", k, tuple(state_dict[k].shape))
    return state_dict


def hf_verify_tokenizer():
    """[4.3] tokenizer_config.json + vocab.json + *.spm → 文本↔id。"""
    import json
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download

    print("=" * 72)
    print("[4.3] vocab.json + source.spm + target.spm → 文本 ↔ id")
    tok = AutoTokenizer.from_pretrained(model_id)
    print("tokenizer 实际类 :", type(tok).__name__)
    print("tokenizer vocab  :", tok.vocab_size)
    print("特殊 token       : pad=", repr(tok.pad_token), tok.pad_token_id,
          "| eos=", repr(tok.eos_token), tok.eos_token_id,
          "| unk=", repr(tok.unk_token), tok.unk_token_id)

    vocab = json.load(open(hf_hub_download(model_id, "vocab.json"), encoding="utf-8"))
    print("vocab.json 条目数:", len(vocab), "| id 范围:",
          min(vocab.values()), "->", max(vocab.values()))
    print("vocab 映射示例   :", {p: vocab.get(p) for p in
                                 ["</s>", "<unk>", "▁the", "的", "。", ">>cmn_Hans<<"]})

    try:
        import sentencepiece as spm
        for name in ("source.spm", "target.spm"):
            sp = spm.SentencePieceProcessor()
            sp.Load(hf_hub_download(model_id, name))
            print(f"{name}: piece_size={sp.GetPieceSize()}")
    except ImportError:
        print("(未安装 sentencepiece，跳过 .spm 细节)")

    text = "How are you?"
    enc = tok([text], return_tensors="pt")
    print("输入文本         :", text)
    print("source.spm 切分  :", tok.convert_ids_to_tokens(enc["input_ids"][0]))
    print("→ input_ids      :", enc["input_ids"].tolist(),
          "（经 vocab.json 映射，非 spm 自身 id）")
    print("→ attention_mask :", enc["attention_mask"].tolist())
    return tok


def hf_verify_generation_config():
    """[4.4] generation_config.json → generate() 的默认策略。"""
    from transformers import GenerationConfig

    print("=" * 72)
    print("[4.4] generation_config.json → generate() 的默认策略")
    gen_cfg = GenerationConfig.from_pretrained(model_id)
    print("num_beams              :", gen_cfg.num_beams)
    print("max_length             :", gen_cfg.max_length)
    print("eos_token_id           :", gen_cfg.eos_token_id)
    print("pad_token_id           :", gen_cfg.pad_token_id)
    print("decoder_start_token_id :", gen_cfg.decoder_start_token_id)
    print("bad_words_ids          :", gen_cfg.bad_words_ids)
    return gen_cfg


def hf_verify_end_to_end():
    """[4.6] 端到端：编码 → 生成 → 解码。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print("=" * 72)
    print("[4.6] 端到端：How are you? → 你好吗?")
    tok = AutoTokenizer.from_pretrained(model_id)
    print(tok)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
    print(model)

    text = "How are you?"
    enc = tok([text], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**enc, max_new_tokens=64)
    print("input_ids     :", enc["input_ids"].tolist())
    print("generated_ids :", generated_ids.tolist())
    print("生成 token    :",
          tok.convert_ids_to_tokens(generated_ids[0], skip_special_tokens=True))
    print("最终输出      :",
          tok.batch_decode(generated_ids, skip_special_tokens=True)[0])
    return generated_ids


def hf_run_all_verifications():
    print("\n" + "#" * 72)
    print("# Hugging Face 模型文件内部结构验证（文档第四节可复现实测）")
    print("#" * 72)
    hf_verify_config()
    hf_verify_weights()
    hf_verify_tokenizer()
    hf_verify_generation_config()
    hf_verify_end_to_end()
    print("=" * 72)
    print("验证完成。")


if __name__ == "__main__":
    print("英译中翻译器已启动，输入英文句子进行翻译（输入 exit / quit / q 退出）：")

    # hf_run_all_verifications()

    while True:
        try:
            text = input("\n请输入英文: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not text:
            continue

        if text.lower() in ("exit", "quit", "q"):
            print("已退出。")
            break

        result = translate(text)
        print("中文翻译:", result)
