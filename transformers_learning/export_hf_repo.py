#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 train_main.py 训练出的 PyTorch 权重导出成**可以直接上传到 HuggingFace Hub、
别人 clone 下来就能跑**的模型仓库（HuggingFace 官方规范：config.json + auto_map +
model.safetensors + tokenizer 文件 + ONNX + 模型卡）。

产出目录（默认 <ckpt 所在目录>/hf_repo）：

    README.md                              模型卡（YAML front matter：library_name / pipeline_tag）
    config.json                            PretrainedConfig，含 auto_map（trust_remote_code）
    generation_config.json                 GenerationConfig（max_length / num_beams / ...）
    model.safetensors                      PyTorch 权重（键名与本工程的 tf_model.py 完全一致）
    configuration_transformer_custom.py   自定义配置类（HF 官方推荐的自定义架构发布方式）
    modeling_transformer_custom.py        自定义模型类（PreTrainedModel，支持 generate()）
    tokenization_transformer_custom.py    自定义分词器（源/目标词表分离 + 句首 BOS）
    tokenizer_config.json                  分词器配置（tokenizer_class + auto_map + special tokens）
    vocab.json / target_vocab.json         英文 / 中文词表（token -> id）
    source.spm / target.spm                英文 / 中文 SentencePiece 模型
    encoder_model.onnx                     推理用 ONNX（onnxruntime 可直接跑）
    decoder_model.onnx
    export_meta.json                       本工程附加的导出/校验元信息

为什么这样就能"别人直接运行"：

    1. `trust_remote_code=True` 路径（HF 原生，最推荐）
           from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
           tok = AutoTokenizer.from_pretrained("<repo>", trust_remote_code=True)
           model = AutoModelForSeq2SeqLM.from_pretrained("<repo>", trust_remote_code=True)
           print(tok.batch_decode(model.generate(**tok(["..."], return_tensors="pt"),
                                                 num_beams=3, max_length=60),
                                  skip_special_tokens=True))
       —— 只需要仓库里的这几个文件，不需要本工程的任何代码；生成的 .spm/.json/.py 都在仓库内自包含。
    2. onnxruntime 路径（不需要 transformers）
       AutoTokenizer 仍然可以用（分词器就是 HF 的 MarianTokenizer 家族），
       模型侧按 README 里的 onnxruntime 片段跑 encoder_model.onnx / decoder_model.onnx。
    3. 上传
           huggingface-cli login
           hf upload <user>/<repo> <导出目录>
       或直接用本脚本的 --upload <user>/<repo> 自动创建仓库并上传。

用法（在 transformers_learning 目录下执行）：

    python export_hf_repo.py                                    # 默认用 config.translate_model_path
    python export_hf_repo.py --repo-id your-name/en-zh-base     # 模型卡里写入真实的仓库 id
    python export_hf_repo.py --clean --no-verify
    python export_hf_repo.py --repo-id your-name/en-zh-base --upload your-name/en-zh-base
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time

import torch

# ---------------------------------------------------------------------------
# 保证可以 import 到工程内的 config / model / 其它导出脚本（无论从哪个目录启动）
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config  # noqa: E402
# ONNX 导出/校验逻辑直接复用 export_onnx_hf.py，避免两份实现
from export_onnx_hf import (  # noqa: E402
    DECODER_FILE, ENCODER_FILE, HFDecoderModel, HFEncoderModel, OnnxSeq2SeqConfig,
    build_and_load_model, export_models, load_state_dict, patch_model_for_onnx,
    resolve_path, verify_onnx,
)

logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("export_hf_repo")

TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "hf_template")
TEMPLATE_FILES = ("configuration_transformer_custom.py",
                  "modeling_transformer_custom.py",
                  "tokenization_transformer_custom.py")
META_FILE = "export_meta.json"
GENERATED_FILES = (ENCODER_FILE, DECODER_FILE, "README.md", "config.json",
                   "generation_config.json", "tokenizer_config.json",
                   "special_tokens_map.json", "vocab.json", "target_vocab.json",
                   "source.spm", "target.spm", "model.safetensors",
                   "pytorch_model.bin", META_FILE) + TEMPLATE_FILES

DEFAULT_REPO_ID = "your-name/transformer-en-zh-base"


# ===========================================================================
# 1) 权重
# ===========================================================================
def save_weights(repo_dir: str, state_dict: dict) -> str:
    """保存 model.safetensors（HF 首选格式）；没装 safetensors 时退回 pytorch_model.bin。"""
    if _module_available("safetensors"):
        from safetensors.torch import save_file
        path = os.path.join(repo_dir, "model.safetensors")
        save_file({k: v.contiguous() for k, v in state_dict.items()}, path,
                  metadata={"format": "pt"})
        LOGGER.info("已保存 %s (%.1f MB)", os.path.basename(path),
                    os.path.getsize(path) / 1024 / 1024)
        return os.path.basename(path)
    path = os.path.join(repo_dir, "pytorch_model.bin")
    torch.save(state_dict, path)
    LOGGER.warning("未安装 safetensors，已退回 %s（pip install safetensors 可生成 HF 首选格式）",
                   os.path.basename(path))
    return os.path.basename(path)


def _module_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


# ===========================================================================
# 2) HF 元数据：config.json / generation_config.json / tokenizer 资产
# ===========================================================================
def build_config_json(output_mode: str) -> dict:
    """对齐 transformers.PretrainedConfig；auto_map 让 HF 从仓库里加载自定义代码。"""
    tgt_vocab = int(config.tgt_vocab_size)
    cfg = {
        "architectures": ["TransformerForConditionalGeneration"],
        "model_type": "transformer_custom",
        "auto_map": {
            "AutoConfig": "configuration_transformer_custom.TransformerCustomConfig",
            "AutoModelForSeq2SeqLM":
                "modeling_transformer_custom.TransformerForConditionalGeneration",
        },
        "is_encoder_decoder": True,
        "d_model": int(config.d_model),
        "n_heads": int(config.n_heads),
        "n_layers": int(config.n_layers),
        "d_ff": int(config.d_ff),
        "dropout": float(config.dropout),
        "src_vocab_size": int(config.src_vocab_size),
        "tgt_vocab_size": tgt_vocab,
        "vocab_size": tgt_vocab,
        "max_position_embeddings": 5000,
        "activation_function": "relu",
        "pre_norm": False,          # Post-LN: x + dropout(sublayer(norm(x)))
        "scale_embedding": True,    # embedding 乘 sqrt(d_model)
        "tie_word_embeddings": False,
        "pad_token_id": int(config.padding_idx),
        "unk_token_id": 1,
        "bos_token_id": int(config.bos_idx),
        "eos_token_id": int(config.eos_idx),
        "decoder_start_token_id": int(config.bos_idx),
        # 本工程附加信息
        "output_mode": output_mode,
    }
    if _module_available("transformers"):
        import transformers
        cfg["transformers_version"] = transformers.__version__
    return cfg


def build_generation_config_json() -> dict:
    return {
        "bos_token_id": int(config.bos_idx),
        "eos_token_id": int(config.eos_idx),
        "pad_token_id": int(config.padding_idx),
        "decoder_start_token_id": int(config.bos_idx),
        "max_length": int(config.max_len),
        "min_length": 0,
        "num_beams": int(config.beam_size),
        # 本模型没实现 KV Cache：显式关掉，避免 transformers 准备 DynamicCache
        "use_cache": False,
        "num_return_sequences": 1,
        "early_stopping": True,
        "length_penalty": 1.0,
        "do_sample": False,
        "repetition_penalty": 1.0,
    }


def build_tokenizer_config_json() -> dict:
    """tokenizer_config.json：tokenizer_class 保底 + auto_map 指向仓库内的自定义分词器。"""
    return {
        "tokenizer_class": "MarianTokenizer",
        "auto_map": {
            "AutoTokenizer": ["tokenization_transformer_custom.TransformerCustomTokenizer", None],
        },
        "source_spm": "source.spm",
        "target_spm": "target.spm",
        "vocab_file": "vocab.json",
        "target_vocab_file": "target_vocab.json",
        # 英中两套独立词表：encode 用 source.spm，decode 用 target.spm
        "separate_vocabs": True,
        "source_lang": "en",
        "target_lang": "zh",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token_id": 1,
        "bos_token_id": int(config.bos_idx),
        "eos_token_id": int(config.eos_idx),
        "pad_token_id": int(config.padding_idx),
        "model_max_length": int(config.max_len),
        "clean_up_tokenization_spaces": True,
    }


def build_special_tokens_map_json() -> dict:
    return {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }


def write_tokenizer_assets(repo_dir: str, src_spm: str, tgt_spm: str) -> None:
    """复制 source.spm / target.spm，并按 id 顺序导出 vocab.json / target_vocab.json。"""
    import sentencepiece as spm

    shutil.copyfile(src_spm, os.path.join(repo_dir, "source.spm"))
    shutil.copyfile(tgt_spm, os.path.join(repo_dir, "target.spm"))

    for model_path, out_name in ((src_spm, "vocab.json"), (tgt_spm, "target_vocab.json")):
        proc = spm.SentencePieceProcessor()
        proc.Load(model_path)
        vocab = {proc.id_to_piece(i): i for i in range(proc.GetPieceSize())}
        with open(os.path.join(repo_dir, out_name), "w", encoding="utf-8") as fp:
            json.dump(vocab, fp, ensure_ascii=False)
        LOGGER.info("已写出 %s（%d 个 piece）", out_name, len(vocab))


# ===========================================================================
# 3) 模型卡
# ===========================================================================
def build_model_card(repo_id: str, ckpt_name: str, generation: dict,
                     onnx_files: list, verified: dict) -> str:
    max_length = generation["max_length"]
    num_beams = generation["num_beams"]
    bos, eos, pad = int(config.bos_idx), int(config.eos_idx), int(config.padding_idx)
    d_model, n_heads, n_layers = int(config.d_model), int(config.n_heads), int(config.n_layers)
    verified_line = ""
    if verified.get("greedy_match") is not None:
        verified_line = (f"\n> 已在本机校验：`transformers`（trust_remote_code）贪心解码与 ONNX 贪心解码"
                         f"逐 token 一致 = {verified['greedy_match']}；"
                         f"HF 与 ONNX 的 logits 最大绝对误差 = {verified.get('max_abs_logits_diff')}\n")
    onnx_names = ", ".join(f"`{os.path.basename(item['file'])}`" for item in onnx_files) or "（未导出）"
    return f"""---
library_name: transformers
pipeline_tag: translation
language:
- en
- zh
tags:
- translation
- onnx
- sentencepiece
- custom-code
---

# Transformer (en → zh) — {d_model}d / {n_layers} layers / {n_heads} heads

基于 **标准 Transformer（Post-LN + 固定 sin-cos 位置编码）** 的英译中模型，由 `train_main.py`
在自建平行语料上训练得到（`{ckpt_name}`，BLEU ≈ 26），本仓库由 `export_hf_repo.py` 导出：

* `model.safetensors`：PyTorch 权重（键名与本工程 `model/tf_model.py` 完全一致）
* `encoder_model.onnx` / `decoder_model.onnx`：**只含推理图**的动态 shape ONNX
* `config.json` + `auto_map`：HuggingFace 官方推荐的"自定义架构"发布方式，配合 `trust_remote_code=True`
* 分词器：英文 / 中文各自一套 SentencePiece（`source.spm` / `target.spm`）

> 结构不是 BART/Marian 等 HF 内置架构，所以模型侧需要 `trust_remote_code=True`（代码随仓库一起分发，
> 见 `modeling_transformer_custom.py` / `configuration_transformer_custom.py` /
> `tokenization_transformer_custom.py`）；ONNX 分支则只依赖 `onnxruntime`。
{verified_line}
## 文件清单

| 文件 | 说明 |
| --- | --- |
| `config.json` | 结构配置（`d_model={d_model}`, `n_layers={n_layers}`, `n_heads={n_heads}`）+ `auto_map` |
| `generation_config.json` | 生成默认值：`max_length={max_length}`, `num_beams={num_beams}`, `early_stopping=true` |
| `model.safetensors` | PyTorch 权重 |
| `encoder_model.onnx` | `input_ids(int64,B,S)` + `attention_mask(int64,B,S)` → `last_hidden_state(float32,B,S,{d_model})` |
| `decoder_model.onnx` | `input_ids(int64,B,T)` + `encoder_hidden_states` + `encoder_attention_mask` → `logits(float32,B,T,tgt_vocab)` |
| `source.spm` / `target.spm` | 英文 / 中文 SentencePiece 模型 |
| `vocab.json` / `target_vocab.json` | token → id（HF 分词器格式） |
| `tokenizer_config.json` | `separate_vocabs=true`（编码用英文 spm、解码用中文 spm）+ 句首补 BOS |
| `*_transformer_custom.py` | 自定义 config / modeling / tokenization 代码 |

## 环境依赖

```bash
# 方式一（PyTorch + generate）
pip install "transformers>=4.40" torch sentencepiece sacremoses
# 方式二（ONNX 推理；分词器仍然用 transformers 里的 MarianTokenizer 家族）
pip install onnxruntime sentencepiece "transformers>=4.40" sacremoses
```

## 快速开始

### 方式一：transformers + trust_remote_code（PyTorch，含 `generate()`）

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

repo = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(repo, trust_remote_code=True)

sentences = ["The government has implemented various policies to improve the living standards of its citizens."]
inputs = tokenizer(sentences, return_tensors="pt", padding=True)
out = model.generate(**inputs, max_length={max_length}, num_beams={num_beams})
print(tokenizer.batch_decode(out, skip_special_tokens=True))
# ['政府实施了诸多政策,改善国民生活水平。']
```

### 方式二：onnxruntime（不需要 transformers 的模型代码）

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer          # 分词器仍是 HF 原生实现

tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)
enc = ort.InferenceSession("encoder_model.onnx", providers=["CPUExecutionProvider"])
dec = ort.InferenceSession("decoder_model.onnx", providers=["CPUExecutionProvider"])

input_ids = np.array([tokenizer("The cat is sleeping on the sofa.")["input_ids"]],
                     dtype=np.int64)                       # 已含首尾 {bos} / {eos}
attention_mask = (input_ids != {pad}).astype(np.int64)
memory = enc.run(None, {{"input_ids": input_ids,
                         "attention_mask": attention_mask}})[0]

cur = np.array([[{bos}]], dtype=np.int64)                   # decoder_start_token_id
for _ in range({max_length}):
    logits = dec.run(None, {{"input_ids": cur,
                             "encoder_hidden_states": memory,
                             "encoder_attention_mask": attention_mask}})[0]
    nxt = int(logits[0, -1].argmax())
    cur = np.concatenate([cur, [[nxt]]], axis=1)
    if nxt == {eos}:
        break
print(tokenizer.batch_decode(cur[:, 1:], skip_special_tokens=True))
```

> 注意：`tokenizer(...)` 会给源句加上 `BOS({bos})` … `EOS({eos})`（本仓库的自定义分词器负责补 BOS，
> 与训练时完全一致）。如果**不**使用 `trust_remote_code=True`，HF 会退回到内置的 `MarianTokenizer`，
> 它只补 EOS、且解码会用英文 spm，结果会明显变差 —— 请务必带上 `trust_remote_code=True`。

### 方式三：把 ONNX 换成 optimum / onnxruntime-genai

本仓库的 ONNX 输入输出名与 optimum 导出的 seq2seq 模型一致
（`input_ids` / `attention_mask` / `encoder_hidden_states` / `encoder_attention_mask` →
`last_hidden_state` / `logits`），可以直接喂给只认 ONNX 文件的推理框架。

## 训练细节

| 项目 | 值 |
| --- | --- |
| 架构 | Transformer encoder-decoder（Post-LN，x + dropout(sublayer(norm(x)))） |
| d_model / heads / layers / d_ff | {d_model} / {n_heads} / {n_layers} / {int(config.d_ff)} |
| dropout | {config.dropout} |
| 位置编码 | 固定 sin-cos（`pe` buffer，max_len 5000） |
| 分词 | SentencePiece BPE，源/目标各 32k，special ids: pad={pad}, unk=1, bos={bos}, eos={eos} |
| 训练数据 | 通用英中平行语料（见原工程 `dataset/`） |
| BLEU | ≈ 26（beam size {num_beams}） |
| 直接解码 | 逐步自回归，未实现 KV Cache（`decoder_with_past_model.onnx` 不适用） |

## 已知限制

* 逐句自回归推理（无 KV Cache），长句偏慢；ONNX 模型 `batch` / `sequence_length` 均为动态维。
* 训练语料偏新闻/书面语，口语与生僻领域效果会下降。
* 生成质量以 beam search（`num_beams={num_beams}`）优于贪心。
* 请根据自己的场景补充许可证（本卡未附带 LICENSE）。

## 上传 / 复现（本仓库的导出方式）

```bash
# 在本工程 transformers_learning 目录下
python export_hf_repo.py --repo-id {repo_id} --clean
hf upload {repo_id} data/train/exp/weights/hf_repo          # 或用 --upload {repo_id}
```

推理图（ONNX）：{onnx_names}
"""


# ===========================================================================
# 4) 仓库自检（模拟"别人拿到仓库后"的加载与推理）
# ===========================================================================
def verify_hf_repo(repo_dir: str, sentences, max_len: int, beam_size: int,
                   atol: float = 1e-3) -> dict:
    import numpy as np
    import sentencepiece as spm

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from translate_onnx_hf import Seq2SeqOnnxModel, cut_at_eos

    report: dict = {"repo_dir": repo_dir}

    # ---- 1. 分词器：HF 加载出来的 id 必须与训练侧人工编码一致 ----
    tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
    report["tokenizer_class"] = type(tokenizer).__name__
    source_sp = spm.SentencePieceProcessor()
    source_sp.Load(os.path.join(repo_dir, "source.spm"))

    # ---- 2. 模型：自定义代码 + safetensors ----
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_dir, trust_remote_code=True)
    model.eval()
    report["model_class"] = type(model).__name__
    n_params = sum(p.numel() for p in model.parameters())
    report["num_parameters"] = int(n_params)
    LOGGER.info("[自检] %s / %s，参数量 %.1fM", report["tokenizer_class"], report["model_class"],
                n_params / 1e6)

    # ---- 3. 逐句：分词一致性 + HF 生成 vs ONNX 解码 + logits 数值对比 ----
    onnx_model = Seq2SeqOnnxModel.from_pretrained(repo_dir, verbose=False)
    per_sentence = []
    all_greedy_matched = True
    max_abs_diff = 0.0

    for sent in sentences:
        inputs = tokenizer([sent], return_tensors="pt")
        manual_ids = [[int(config.bos_idx)] + source_sp.EncodeAsIds(sent) + [int(config.eos_idx)]]
        ids_match = inputs["input_ids"].tolist() == manual_ids

        with torch.no_grad():
            hf_greedy = model.generate(**inputs, max_length=max_len, num_beams=1, do_sample=False)
            hf_beam = (model.generate(**inputs, max_length=max_len, num_beams=beam_size)
                       if beam_size > 1 else None)

        memory, attention_mask, _ = onnx_model.encode([sent])
        onnx_greedy = onnx_model.greedy_decode(memory, attention_mask, max_len)
        onnx_beam = (onnx_model.beam_search(memory, attention_mask, max_len, beam_size)
                     if beam_size > 1 else None)

        eos_idx = int(config.eos_idx)
        greedy_matched = (cut_at_eos(hf_greedy[0].tolist(), eos_idx)
                          == cut_at_eos(onnx_greedy, eos_idx))
        all_greedy_matched = all_greedy_matched and greedy_matched

        # HF(PyTorch) 与 ONNX 的 logits 直接比数值
        with torch.no_grad():
            hf_logits = model(input_ids=inputs["input_ids"],
                              attention_mask=inputs["attention_mask"],
                              decoder_input_ids=torch.tensor([onnx_greedy])).logits.numpy()
        ort_logits = onnx_model.decoder.run(None, {
            "input_ids": np.array([onnx_greedy], dtype=np.int64),
            "encoder_hidden_states": memory,
            "encoder_attention_mask": attention_mask,
        })[0]
        diff = float(np.abs(hf_logits - ort_logits).max())
        max_abs_diff = max(max_abs_diff, diff)

        item = {
            "sentence": sent,
            "tokenizer_ids_match": bool(ids_match),
            "hf_greedy_text": tokenizer.batch_decode(hf_greedy, skip_special_tokens=True)[0],
            "onnx_greedy_text": onnx_model.decode_text(onnx_greedy),
            "greedy_token_match": bool(greedy_matched),
            "max_abs_logits_diff": diff,
            "logits_allclose": bool(np.allclose(hf_logits, ort_logits, atol=atol, rtol=atol)),
        }
        if hf_beam is not None:
            item["hf_beam_text"] = tokenizer.batch_decode(hf_beam, skip_special_tokens=True)[0]
            item["onnx_beam_text"] = onnx_model.decode_text(onnx_beam)
        per_sentence.append(item)

        LOGGER.info("[自检] %s", sent[:60])
        LOGGER.info("[自检]   分词一致=%s | 贪心逐 token 一致=%s | logits max_abs_diff=%.3e",
                    ids_match, greedy_matched, diff)
        LOGGER.info("[自检]   HF : %s", item["hf_greedy_text"])
        LOGGER.info("[自检]   ONNX: %s", item["onnx_greedy_text"])
        if "hf_beam_text" in item:
            LOGGER.info("[自检]   HF beam  : %s", item["hf_beam_text"])
            LOGGER.info("[自检]   ONNX beam: %s", item["onnx_beam_text"])

    report["sentences"] = per_sentence
    report["greedy_match"] = bool(all_greedy_matched)
    report["max_abs_logits_diff"] = max_abs_diff
    return report


# ===========================================================================
# 5) 主流程
# ===========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="export_hf_repo.py",
        description="导出可直接上传 HuggingFace Hub 的模型仓库（safetensors + ONNX + 自定义代码 + 模型卡）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", default=config.translate_model_path,
                        help="PyTorch 权重路径，默认取 config.translate_model_path")
    parser.add_argument("--repo-dir", default=None, help="导出目录，默认 <ckpt目录>/hf_repo")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                        help="HuggingFace 仓库 id（写进模型卡示例，如 your-name/en-zh-base）")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset，<=0 表示交给 PyTorch 自动选择")
    parser.add_argument("--exporter", choices=["auto", "dynamo", "legacy"], default="auto",
                        help="ONNX 导出后端")
    parser.add_argument("--patch", choices=["auto", "always", "never"], default="auto",
                        help="ONNX 友好化补丁：auto=失败时自动打")
    parser.add_argument("--output", choices=["logits", "log_probs"], default="logits",
                        help="decoder 输出：HF 约定的 logits 或原 Generator 的 log_probs")
    parser.add_argument("--batch-size", type=int, default=2, help="导出用 dummy batch size")
    parser.add_argument("--src-len", type=int, default=16, help="导出用 dummy 源句长度")
    parser.add_argument("--tgt-len", type=int, default=16, help="导出用 dummy 目标句长度")
    parser.add_argument("--no-onnx", dest="onnx", action="store_false", help="只导出 HF 仓库，不导出 ONNX")
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="跳过导出后校验")
    parser.add_argument("--atol", type=float, default=1e-3, help="数值校验绝对容差")
    parser.add_argument("--rtol", type=float, default=1e-3, help="数值校验相对容差")
    parser.add_argument("--seed", type=int, default=0, help="dummy 输入随机种子")
    parser.add_argument("--clean", action="store_true", help="导出前清理目标目录旧产物")
    parser.add_argument("--coreml", action="store_true", help="校验 ONNX 时使用 CoreML EP")
    parser.add_argument("--from-dev", type=int, default=2, help="自检用 dev.json 前 N 句")
    parser.add_argument("--text", action="append", default=None, help="自检用英文句子，可重复指定")
    parser.add_argument("--upload", default=None,
                        help="导出后直接上传到该仓库 id（需要 huggingface-cli login / HF_TOKEN）")
    parser.add_argument("--private", action="store_true", help="把仓库创建为 private")
    parser.add_argument("--hub-token", default=os.environ.get("HF_TOKEN"),
                        help="HF token，默认读环境变量 HF_TOKEN")
    parser.add_argument("--no-upload-verify", dest="upload_verify", action="store_false",
                        help="上传后不做「下载回本地重新校验」（默认会做，需再下载约 790MB）")
    parser.add_argument("--upload-verify-dir", default=None,
                        help="上传校验时下载到哪个本地目录（默认用 HF 缓存）")
    return parser.parse_args(argv)


def load_check_sentences(args) -> list:
    if args.text:
        return list(args.text)
    path = os.path.join(SCRIPT_DIR, config.dev_data_path.lstrip("./"))
    if not os.path.isfile(path):
        return ["The cat is sleeping on the sofa."]
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    return [row[0] for row in data[: max(1, args.from_dev)]]


def upload_repo(repo_dir: str, repo_id: str, args, sentences: list) -> dict:
    """上传到 HuggingFace Hub，并做「上传后校验」。

    校验方式：从 Hub 下载回本地 -> 逐文件哈希比对（大文件 lfs.sha256 / 小文件 git blob sha1）
    -> ONNX 推理 -> transformers(trust_remote_code) 推理 -> 与本地 PyTorch 原模型逐 token 对照。
    返回可直接写进 export_meta.json 的 dict。
    """
    from huggingface_hub import HfApi, create_repo

    token = args.hub_token or os.environ.get("HF_TOKEN")
    files = [name for name in sorted(os.listdir(repo_dir))
             if os.path.isfile(os.path.join(repo_dir, name))]
    total_mb = sum(os.path.getsize(os.path.join(repo_dir, name)) for name in files) / 1024 / 1024
    LOGGER.info("开始上传 %s（%d 个文件，%.1f MB）", repo_id, len(files), total_mb)

    create_repo(repo_id, repo_type="model", exist_ok=True, private=bool(args.private), token=token)
    started = time.time()
    commit = HfApi(token=token).upload_folder(
        repo_id=repo_id, folder_path=repo_dir, token=token,
        commit_message="Upload transformer en-zh (safetensors + ONNX + custom code)")
    duration = time.time() - started

    report = {
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/{repo_id}",
        "private": bool(args.private),
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_files": len(files),
        "total_mb": round(total_mb, 2),
        "duration_s": round(duration, 1),
        "commit_revision": getattr(commit, "oid", None) or HfApi(token=token).model_info(repo_id).sha,
    }
    LOGGER.info("上传完成（%.1fs）: %s", duration, report["repo_url"])

    if not args.upload_verify:
        LOGGER.info("已跳过上传后校验（--no-upload-verify）")
        return report

    from translate_onnx_hf import verify_hub_download

    LOGGER.info("[上传校验] 从 Hub 下载回来重新校验：逐文件哈希 + HF/ONNX/本地 PyTorch 逐 token 对照 ...")
    verification = verify_hub_download(
        repo_id, sentences, max_len=int(config.max_len), beam_size=int(config.beam_size),
        revision=report["commit_revision"], download_dir=args.upload_verify_dir, token=token,
        ckpt=resolve_path(args.ckpt), with_transformers=True, coreml=args.coreml)
    report["verification"] = verification
    report["verified"] = bool(verification["all_matched"])
    if report["verified"]:
        LOGGER.info("[上传校验] PASS ✅ %s 可加载、可推理，且与本地 PyTorch 逐 token 一致", report["repo_url"])
    else:
        LOGGER.error("[上传校验] FAIL ❌ 校验未通过，请查看上面的明细")
    return report


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    ckpt_path = resolve_path(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")
    repo_dir = os.path.abspath(resolve_path(
        args.repo_dir or os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "hf_repo")))
    # 模型卡里写真实仓库 id：--upload 时若没显式指定 --repo-id，就沿用上传目标
    if args.upload and args.repo_id == DEFAULT_REPO_ID:
        args.repo_id = args.upload
        LOGGER.info("模型卡使用仓库 id: %s", args.repo_id)
    os.makedirs(repo_dir, exist_ok=True)
    if args.clean:
        for name in GENERATED_FILES:
            target = os.path.join(repo_dir, name)
            if os.path.isfile(target):
                os.remove(target)
        LOGGER.info("已清理目标目录旧产物: %s", repo_dir)

    # ---- 模型 & 权重（eval + no_grad，只做推理）----
    model = build_and_load_model(ckpt_path, device="cpu")
    padding_idx = int(config.padding_idx)
    state_dict = load_state_dict(ckpt_path, map_location="cpu")

    exported, verification = [], {}
    if args.onnx:
        onnx_config = OnnxSeq2SeqConfig(args, args.output, padding_idx)
        input_ids, attention_mask, decoder_input_ids = onnx_config.dummy_inputs(
            max(1, args.batch_size), max(1, args.src_len), max(2, args.tgt_len))
        encoder_module = HFEncoderModel(model).eval()
        decoder_module = HFDecoderModel(model, padding_idx, args.output).eval()

        def _apply_patch() -> None:
            with torch.no_grad():
                before = decoder_module(decoder_input_ids,
                                        encoder_module(input_ids, attention_mask), attention_mask)
            patch_model_for_onnx()
            with torch.no_grad():
                after = decoder_module(decoder_input_ids,
                                       encoder_module(input_ids, attention_mask), attention_mask)
            LOGGER.info("补丁等价性检查: max_abs_diff=%.3e", float((after - before).abs().max()))

        import inspect
        dynamo_supported = "dynamo" in inspect.signature(torch.onnx.export).parameters
        patch_applied = args.patch == "always"
        if patch_applied:
            _apply_patch()
        try:
            exported = export_models(encoder_module, decoder_module, input_ids, attention_mask,
                                     decoder_input_ids, onnx_config, repo_dir, args,
                                     dynamo_supported)
        except RuntimeError as exc:
            if patch_applied or args.patch == "never":
                raise
            LOGGER.warning("未打补丁导出失败，改用补丁后重试: %s", str(exc)[:200])
            _apply_patch()
            exported = export_models(encoder_module, decoder_module, input_ids, attention_mask,
                                     decoder_input_ids, onnx_config, repo_dir, args,
                                     dynamo_supported)
        if args.verify:
            verification = verify_onnx(os.path.join(repo_dir, ENCODER_FILE),
                                       os.path.join(repo_dir, DECODER_FILE),
                                       encoder_module, decoder_module,
                                       input_ids, attention_mask, decoder_input_ids, args)

    # ---- 权重 / 元数据 / 分词器 / 自定义代码 / 模型卡 ----
    weight_file = save_weights(repo_dir, state_dict)
    generation = build_generation_config_json()
    for name, payload in (("config.json", build_config_json(args.output)),
                          ("generation_config.json", generation),
                          ("tokenizer_config.json", build_tokenizer_config_json()),
                          ("special_tokens_map.json", build_special_tokens_map_json())):
        with open(os.path.join(repo_dir, name), "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    src_spm = resolve_path("tokenizer/eng.model")
    tgt_spm = resolve_path("tokenizer/chn.model")
    write_tokenizer_assets(repo_dir, src_spm, tgt_spm)

    for name in TEMPLATE_FILES:
        shutil.copyfile(os.path.join(TEMPLATE_DIR, name), os.path.join(repo_dir, name))

    # ---- 自检：模拟别人 trust_remote_code 加载 + generate ----
    hf_check = {}
    if args.verify:
        try:
            hf_check = verify_hf_repo(repo_dir, load_check_sentences(args),
                                      int(config.max_len), int(config.beam_size), args.atol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("[自检] HF 仓库加载/generate 失败: %s: %s", type(exc).__name__, exc)
            raise

    with open(os.path.join(repo_dir, "README.md"), "w", encoding="utf-8") as fp:
        fp.write(build_model_card(args.repo_id, os.path.basename(ckpt_path), generation,
                                  exported, hf_check))

    LOGGER.info("导出完成: %s", repo_dir)
    for name in sorted(os.listdir(repo_dir)):
        size = os.path.getsize(os.path.join(repo_dir, name)) / 1024 / 1024
        LOGGER.info("  - %-40s %8.2f MB", name, size)

    # ---- 上传 + 上传后校验（下载回本地重新验证），报告写进 export_meta.json ----
    upload_report = {}
    if args.upload:
        try:
            upload_report = upload_repo(repo_dir, args.upload, args, load_check_sentences(args))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("上传/上传校验失败（可手动执行 hf upload %s %s）: %s: %s",
                         args.upload, repo_dir, type(exc).__name__, exc)
            upload_report = {"repo_id": args.upload,
                             "repo_url": f"https://huggingface.co/{args.upload}",
                             "error": f"{type(exc).__name__}: {exc}"}
    else:
        LOGGER.info("未指定 --upload，跳过上传；手动上传: huggingface-cli login && hf upload %s %s",
                    args.repo_id, repo_dir)

    meta_path = os.path.join(repo_dir, META_FILE)
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump({
            "checkpoint": ckpt_path,
            "repo_id": args.repo_id,
            "weight_file": weight_file,
            "onnx_files": exported,
            "onnx_verification": verification,
            "hf_repo_verification": hf_check,
            "generation_config": generation,
            "inference_only": True,
            "upload": upload_report,
        }, fp, ensure_ascii=False, indent=2)

    # 把带「上传校验结论」的 export_meta.json 同步回仓库（只传这个几 KB 的文件）
    if upload_report.get("repo_id") and not upload_report.get("error"):
        token = args.hub_token or os.environ.get("HF_TOKEN")
        try:
            from huggingface_hub import HfApi
            HfApi(token=token).upload_file(
                path_or_fileobj=meta_path, path_in_repo=META_FILE,
                repo_id=upload_report["repo_id"], token=token,
                commit_message="Sync export_meta.json (upload verification report)")
            LOGGER.info("已把上传校验报告同步进仓库的 %s: %s", META_FILE,
                        HfApi(token=token).model_info(upload_report["repo_id"]).sha[:12])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("同步 %s 失败（模型文件不受影响）: %s: %s",
                           META_FILE, type(exc).__name__, exc)

    if not args.upload:
        return 0
    if upload_report.get("error"):
        return 1
    return 0 if upload_report.get("verified", True) else 1


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
