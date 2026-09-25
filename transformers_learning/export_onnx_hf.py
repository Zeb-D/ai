#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 train_main.py 训练出的 PyTorch 权重（.pth）按 HuggingFace / Optimum 的规范导出成 ONNX，
**只导出推理图**（eval + no_grad + TrainingMode.EVAL，不含 loss / 梯度 / dropout 相关的训练产物）。

导出目录（默认 <ckpt 所在目录>/onnx_hf）完全对齐 HuggingFace 模型仓库的结构：

    config.json               # 类似 transformers.PretrainedConfig：模型结构 + special token id
    generation_config.json    # 类似 transformers.GenerationConfig：max_length / num_beams / ...
    tokenizer_config.json     # 类似 tokenizer 配置：sentencepiece + special tokens
    source.spm / target.spm   # 由 tokenizer/eng.model、tokenizer/chn.model 复制而来（目录自包含）
    encoder_model.onnx        # 文件名与 optimum 导出的 seq2seq 模型一致
    decoder_model.onnx
    README.md                 # 模型卡（library_name / pipeline_tag 等 front matter，与 HF 仓库一致）
    export_meta.json          # 本工程附加的导出元信息（opset、校验结果等，HF 无此文件）

图和 optimum 导出的 seq2seq 模型同名、同输入输出约定：

    encoder_model.onnx
        inputs : input_ids            int64 (batch, sequence_length)
                 attention_mask       int64 (batch, sequence_length)   真实 token=1，padding=0
        output : last_hidden_state    float32 (batch, sequence_length, d_model)

    decoder_model.onnx
        inputs : input_ids            int64 (batch, sequence_length)
                 encoder_hidden_states    float32 (batch, encoder_sequence_length, d_model)
                 encoder_attention_mask   int64 (batch, encoder_sequence_length)
        output : logits               float32 (batch, sequence_length, tgt_vocab_size)

设计说明：
    1. 只导出推理：模型 .eval()、导出时 torch.no_grad()、旧导出器额外指定 TrainingMode.EVAL；
       包装模块的 forward 只返回张量（HF 风格），不返回 loss / attention 权重等训练期产物。
    2. 不导出 decoder_with_past_model.onnx（KV Cache 增量解码）：本工程的
       MultiHeadedAttention 每步都重算整段 self-attention，没有 past_key_values 结构，
       HF 的 decoder_with_past 只有在模型支持 cache 时才有意义，所以这里只给"整段前向"的推理图。
    3. mask 全部在图内构造，规则与训练侧 tools/data_loader.py 完全一致：
           src_mask = (attention_mask != 0).unsqueeze(-2)                     -> (B, 1, S)
           tgt_mask = (input_ids != padding_idx).unsqueeze(-2) & tril(...)     -> (B, T, T)
       attention_mask 语义采用 HF 约定（真实 token=1），因此从 mask 反推，而不是拿 id 和 pad 比较。
    4. 输出默认是 HF 约定的 logits（不做 log_softmax）；想要原 Generator 的 log_probs 加 --output log_probs。
    5. 导出后会用 onnxruntime 做三项检查：onnx.checker 结构校验、与 PyTorch 的逐值数值对比、
       换一个 batch/序列长度再跑一次（确认 batch / sequence_length 真的是动态维）。

用法（在 transformers_learning 目录下执行）：
    python export_onnx_hf.py                                        # 默认导出 config.translate_model_path
    python export_onnx_hf.py --ckpt data/train/exp/weights/best_bleu_26.30.pth
    python export_onnx_hf.py --output log_probs --out-dir ./onnx_hf
    python export_onnx_hf.py --demo --text "The cat is sleeping on the sofa."

导出后端依赖：
    新导出器（默认，torch.export）: pip install onnx onnxscript
    旧导出器（--exporter legacy） : pip install onnx
    数值/动态轴校验（可选）        : pip install onnxruntime

导出后的纯 onnxruntime 推理示例（贪心解码）：

    import numpy as np
    import onnxruntime as ort

    enc = ort.InferenceSession("encoder_model.onnx", providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession("decoder_model.onnx", providers=["CPUExecutionProvider"])

    input_ids = np.array([[2, 105, 408, 2563, 3]], dtype=np.int64)   # <s> ... </s>
    attention_mask = np.ones_like(input_ids)
    memory = enc.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]

    cur = np.array([[2]], dtype=np.int64)                            # decoder_start_token_id
    for _ in range(60):
        logits = dec.run(None, {"input_ids": cur,
                                "encoder_hidden_states": memory,
                                "encoder_attention_mask": attention_mask})[0]
        nxt = int(logits[0, -1].argmax())
        cur = np.concatenate([cur, [[nxt]]], axis=1)
        if nxt == 3:                                                 # eos
            break
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import logging
import os
import shutil
import sys

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 保证可以 import 到工程内的 config / model（无论从哪个目录启动脚本）
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config  # noqa: E402
from model import tf_model  # noqa: E402
from model.tf_model import make_model  # noqa: E402

logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("export_onnx_hf")

ENCODER_FILE = "encoder_model.onnx"
DECODER_FILE = "decoder_model.onnx"
META_FILE = "export_meta.json"
GENERATED_FILES = (
    ENCODER_FILE, DECODER_FILE,
    "config.json", "generation_config.json", "tokenizer_config.json",
    META_FILE, "README.md", "source.spm", "target.spm",
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _transformers_version() -> str:
    if not _module_available("transformers"):
        return "unknown"
    try:
        import transformers
        return transformers.__version__
    except Exception:  # noqa: BLE001
        return "unknown"


# ===========================================================================
# 1) mask：与训练侧 tools/data_loader.py 完全一致的图内实现
# ===========================================================================
def make_src_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """HF 风格的 attention_mask (B, S) -> (B, 1, S)，供 encoder 的 self-attention 广播使用。"""
    return (attention_mask != 0).unsqueeze(-2)


def make_tgt_mask(input_ids: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """等价 data_loader.Batch.make_std_mask：(B, T) -> (B, T, T)，padding mask & 下三角。"""
    length = input_ids.size(1)
    positions = torch.arange(length, device=input_ids.device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)          # tril（含对角线）
    return (input_ids != padding_idx).unsqueeze(-2) & causal


# ===========================================================================
# 2) ONNX 友好化补丁（数学等价，仅在导出脚本里生效）
# ===========================================================================
def patch_model_for_onnx() -> None:
    """把模型里对 ONNX 导出不友好的实现替换成等价且可导出的写法。"""

    def positional_encoding_forward(self, x):
        # 原实现: x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # torch.autograd.Variable 已废弃，在新导出器（torch.export）下容易报错，直接切片即可。
        return self.dropout(x + self.pe[:, : x.size(1)])

    def layer_norm_forward(self, x):
        # 原实现: std = x.std(-1, keepdim=True)  （无偏标准差，correction=1）
        # 这里用基础算子显式复现：std = sqrt(sum((x - mean)^2) / (N - 1))
        mean = x.mean(-1, keepdim=True)
        n = x.size(-1)
        diff = x - mean
        var = (diff * diff).sum(-1, keepdim=True) / (n - 1)
        return self.a_2 * (x - mean) / torch.sqrt(var + self.eps) + self.b_2

    tf_model.PositionalEncoding.forward = positional_encoding_forward
    tf_model.LayerNorm.forward = layer_norm_forward
    LOGGER.info("已应用 ONNX 友好化补丁: PositionalEncoding.forward / LayerNorm.forward")


# ===========================================================================
# 3) 导出用包装模块（HF 风格：输入输出名 + 图内 mask，只做推理）
# ===========================================================================
class HFEncoderModel(nn.Module):
    """encoder_model.onnx 对应的图：input_ids / attention_mask -> last_hidden_state。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Transformer.encode 内部已经做了 src_embed，这里只传 token id 和 mask
        return self.model.encode(input_ids, make_src_mask(attention_mask))


class HFDecoderModel(nn.Module):
    """decoder_model.onnx 对应的图：input_ids / encoder_hidden_states / encoder_attention_mask -> logits。"""

    def __init__(self, model, padding_idx: int, output_mode: str = "logits"):
        super().__init__()
        self.model = model
        self.padding_idx = int(padding_idx)
        self.output_mode = output_mode

    def forward(self, input_ids: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: torch.Tensor) -> torch.Tensor:
        # Transformer.decode 内部已经做了 tgt_embed，这里同样只传 token id 和 mask
        hidden = self.model.decode(
            encoder_hidden_states,
            make_src_mask(encoder_attention_mask),
            input_ids,
            make_tgt_mask(input_ids, self.padding_idx),
        )
        if self.output_mode == "logits":
            return self.model.generator.proj(hidden)     # HF 约定：未归一化的 logits
        return self.model.generator(hidden)              # 原 Generator：log_softmax 之后的 log_probs


class OnnxSeq2SeqConfig:
    """极简版 optimum.onnx.config.OnnxConfig：只描述导出所需的输入 / 输出 / 动态轴。"""

    def __init__(self, args, output_mode: str, padding_idx: int):
        self.args = args
        self.output_mode = output_mode
        self.padding_idx = int(padding_idx)

    # ---- encoder ----
    @property
    def encoder_input_names(self):
        return ["input_ids", "attention_mask"]

    @property
    def encoder_output_names(self):
        return ["last_hidden_state"]

    @property
    def encoder_dynamic_axes(self):
        axes = {0: "batch", 1: "sequence_length"}
        return {"input_ids": dict(axes), "attention_mask": dict(axes),
                "last_hidden_state": dict(axes)}

    # ---- decoder ----
    @property
    def decoder_input_names(self):
        return ["input_ids", "encoder_hidden_states", "encoder_attention_mask"]

    @property
    def decoder_output_names(self):
        return ["logits" if self.output_mode == "logits" else "log_probs"]

    @property
    def decoder_dynamic_axes(self):
        tgt_axes = {0: "batch", 1: "sequence_length"}
        src_axes = {0: "batch", 1: "encoder_sequence_length"}
        return {"input_ids": dict(tgt_axes),
                "encoder_hidden_states": dict(src_axes),
                "encoder_attention_mask": dict(src_axes),
                self.decoder_output_names[0]: dict(tgt_axes)}

    @property
    def opset(self):
        return int(self.args.opset)

    def dummy_inputs(self, batch_size: int, src_len: int, tgt_len: int):
        """HF 风格：输入 id 落在 [1, vocab) 之间，句尾留一个 padding 以覆盖 mask 分支。"""
        vocab_src = int(config.src_vocab_size)
        vocab_tgt = int(config.tgt_vocab_size)
        input_ids = torch.randint(1, vocab_src, (batch_size, src_len), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        if batch_size >= 2:                       # 最后一行人为造一个 pad，验证 mask 在图内生效
            input_ids[-1, -1] = self.padding_idx
            attention_mask[-1, -1] = 0
        decoder_input_ids = torch.randint(1, vocab_tgt, (batch_size, tgt_len), dtype=torch.long)
        return input_ids, attention_mask, decoder_input_ids


# ===========================================================================
# 4) 权重加载
# ===========================================================================
def resolve_path(path: str) -> str:
    """相对路径优先按"当前工作目录"解析，再按"脚本所在目录"解析。"""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(SCRIPT_DIR, path)
    return candidate if os.path.exists(candidate) else path


def load_state_dict(ckpt_path: str, map_location: str = "cpu") -> dict:
    """加载 .pth；兼容 {'state_dict': ...} 包装与 DataParallel 的 'module.' 前缀。"""
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("weights_only=True 加载失败(%s)，改用 weights_only=False 重试", exc)
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    if isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict):
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise TypeError(f"无法识别的 checkpoint 内容类型: {type(ckpt)}")

    state_dict = {}
    for key, value in ckpt.items():
        if not isinstance(value, torch.Tensor):
            continue
        state_dict[key[7:] if key.startswith("module.") else key] = value
    return state_dict


def build_and_load_model(ckpt_path: str, device: str = "cpu"):
    LOGGER.info("构建模型: d_model=%d, n_heads=%d, n_layers=%d, d_ff=%d, dropout=%.2f, vocab=%d/%d",
                config.d_model, config.n_heads, config.n_layers, config.d_ff, config.dropout,
                config.src_vocab_size, config.tgt_vocab_size)
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model.to(device)                       # 训练可能在 mps/cuda 上完成，ONNX 导出必须在 CPU 上做

    state_dict = load_state_dict(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        LOGGER.warning("严格加载权重失败，改为非严格加载: %s", exc)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            LOGGER.warning("缺失参数(%d个): %s ...", len(missing), list(missing)[:8])
        if unexpected:
            LOGGER.warning("多余参数(%d个): %s ...", len(unexpected), list(unexpected)[:8])
    model.eval()                           # 只做推理：eval() 关掉 dropout 与 BN 的统计更新
    for param in model.parameters():
        param.requires_grad_(False)
    LOGGER.info("权重加载完成: %s (共 %d 个张量)", ckpt_path, len(state_dict))
    return model


# ===========================================================================
# 5) 导出
# ===========================================================================
def _exporter_candidates(preferred: str):
    """返回尝试顺序；True 表示新的 torch.export(dynamo) 导出器。"""
    if preferred == "dynamo":
        return [True, False]
    if preferred == "legacy":
        return [False, True]
    return [True, False] if _module_available("onnxscript") else [False, True]


def _filter_kwargs(func, kwargs: dict) -> dict:
    accepted = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in accepted}


def export_onnx(module, args, path, input_names, output_names, dynamic_axes,
                opset: int, exporter: str = "auto", dynamo_supported: bool = True):
    """按候选后端依次尝试导出，返回真正生效的后端名。"""
    kwargs = _filter_kwargs(torch.onnx.export, {
        "input_names": list(input_names),
        "output_names": list(output_names),
        "dynamic_axes": dict(dynamic_axes),
        "external_data": False,            # 模型 <2GB，强制单文件，避免生成 .onnx.data
        "verbose": False,
        **({"opset_version": int(opset)} if opset and opset > 0 else {}),
    })

    last_error = None
    for dynamo in _exporter_candidates(exporter):
        if dynamo and not dynamo_supported:
            continue
        call_kwargs = dict(kwargs)
        call_kwargs["dynamo"] = dynamo
        if not dynamo:                     # 旧导出器专有参数：显式声明"只导出推理图"
            call_kwargs.update({
                "export_params": True,
                "do_constant_folding": True,
                "training": torch.onnx.TrainingMode.EVAL,
            })
        call_kwargs = _filter_kwargs(torch.onnx.export, call_kwargs)

        backend = "dynamo(torch.export)" if dynamo else "legacy(TorchScript)"
        try:
            with torch.no_grad():          # 只做推理：全程 no_grad
                torch.onnx.export(module, args, path, **call_kwargs)
            LOGGER.info("导出成功 [%s] -> %s (%.1f MB)", backend, path,
                        os.path.getsize(path) / 1024 / 1024)
            return backend
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.warning("使用 %s 导出 %s 失败: %s: %s", backend, os.path.basename(path),
                           type(exc).__name__, str(exc)[:400])

    raise RuntimeError(
        f"ONNX 导出失败: {os.path.basename(path)}，最后一个错误: {last_error}\n"
        "请先安装依赖: pip install onnx onnxscript   # 或缺 onnxscript 时用 --exporter legacy"
    )


def export_models(encoder_module, decoder_module, input_ids, attention_mask, decoder_input_ids,
                  onnx_config: OnnxSeq2SeqConfig, out_dir: str, args, dynamo_supported: bool):
    """导出 encoder_model.onnx + decoder_model.onnx（decoder 的 dummy memory 由 encoder 现场算）。"""
    exported = []

    enc_path = os.path.join(out_dir, ENCODER_FILE)
    enc_backend = export_onnx(
        encoder_module, (input_ids, attention_mask), enc_path,
        input_names=onnx_config.encoder_input_names,
        output_names=onnx_config.encoder_output_names,
        dynamic_axes=onnx_config.encoder_dynamic_axes,
        opset=onnx_config.opset, exporter=args.exporter, dynamo_supported=dynamo_supported,
    )
    exported.append({
        "file": ENCODER_FILE, "backend": enc_backend,
        "inputs": {"input_ids": "int64 (batch, sequence_length)",
                   "attention_mask": "int64 (batch, sequence_length)"},
        "outputs": {"last_hidden_state": "float32 (batch, sequence_length, d_model)"},
    })

    with torch.no_grad():
        memory = encoder_module(input_ids, attention_mask)

    dec_name = onnx_config.decoder_output_names[0]
    dec_path = os.path.join(out_dir, DECODER_FILE)
    dec_backend = export_onnx(
        decoder_module, (decoder_input_ids, memory, attention_mask), dec_path,
        input_names=onnx_config.decoder_input_names,
        output_names=onnx_config.decoder_output_names,
        dynamic_axes=onnx_config.decoder_dynamic_axes,
        opset=onnx_config.opset, exporter=args.exporter, dynamo_supported=dynamo_supported,
    )
    exported.append({
        "file": DECODER_FILE, "backend": dec_backend,
        "inputs": {"input_ids": "int64 (batch, sequence_length)",
                   "encoder_hidden_states": "float32 (batch, encoder_sequence_length, d_model)",
                   "encoder_attention_mask": "int64 (batch, encoder_sequence_length)"},
        "outputs": {dec_name: f"float32 (batch, sequence_length, {config.tgt_vocab_size})"},
    })
    return exported


# ===========================================================================
# 6) 校验：onnx.checker + 数值对比 + 动态轴
# ===========================================================================
def _to_numpy(tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor


def _compare(onnx_out, torch_out, atol: float, rtol: float, tag: str) -> dict:
    import numpy as np
    diff = np.abs(onnx_out - torch_out)
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float((diff / np.maximum(np.abs(torch_out), 1e-6)).max()) if diff.size else 0.0
    passed = bool(np.allclose(onnx_out, torch_out, atol=atol, rtol=rtol))
    LOGGER.info("[校验] %-18s shape=%s max_abs_diff=%.3e max_rel_diff=%.3e 通过=%s",
                tag, list(torch_out.shape), max_abs, max_rel, passed)
    return {"shape": list(torch_out.shape), "max_abs_diff": max_abs,
            "max_rel_diff": max_rel, "passed": passed}


def verify_onnx(enc_path, dec_path, encoder_module, decoder_module,
                input_ids, attention_mask, decoder_input_ids, args) -> dict:
    """导出后校验：结构合法性 / 与 PyTorch 数值一致 / 动态维可用。"""
    report: dict = {}

    if _module_available("onnx"):
        import onnx
        onnx.checker.check_model(enc_path)
        onnx.checker.check_model(dec_path)
        LOGGER.info("[校验] onnx.checker 通过（%s, %s）", ENCODER_FILE, DECODER_FILE)
        report["checker"] = True
    else:
        LOGGER.warning("未安装 onnx，跳过 onnx.checker 结构校验")

    if not _module_available("onnxruntime"):
        LOGGER.warning("未安装 onnxruntime，跳过数值/动态轴校验（pip install onnxruntime 可自动开启）")
        return report

    import numpy as np
    import onnxruntime as ort

    LOGGER.info("onnxruntime %s, providers=%s", ort.__version__, ort.get_available_providers())
    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if args.coreml and "CoreMLExecutionProvider" in available:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    enc = ort.InferenceSession(enc_path, providers=providers)
    dec = ort.InferenceSession(dec_path, providers=providers)

    src = _to_numpy(input_ids)
    mask = _to_numpy(attention_mask)
    tgt = _to_numpy(decoder_input_ids)

    # ---- 数值对比 ----
    memory = enc.run(None, {"input_ids": src, "attention_mask": mask})[0]
    logits = dec.run(None, {"input_ids": tgt, "encoder_hidden_states": memory,
                            "encoder_attention_mask": mask})[0]
    with torch.no_grad():
        ref_memory = _to_numpy(encoder_module(input_ids, attention_mask))
        ref_logits = _to_numpy(decoder_module(decoder_input_ids, encoder_module(input_ids, attention_mask),
                                              attention_mask))
    tag = "logits" if args.output == "logits" else "log_probs"
    report["numerical"] = {
        "encoder.last_hidden_state": _compare(memory, ref_memory, args.atol, args.rtol,
                                              "last_hidden_state"),
        tag: _compare(logits, ref_logits, args.atol, args.rtol, tag),
    }
    # 翻译任务真正关心的是 argmax 出来的 token 是否一致（logits 里大量接近 0 的值会放大相对误差）
    argmax_match = float((logits.argmax(axis=-1) == ref_logits.argmax(axis=-1)).mean())
    report["decoder_argmax_match_ratio"] = argmax_match
    LOGGER.info("[校验] decoder 每步 argmax 一致率: %.6f", argmax_match)

    # ---- 动态轴：换一组 batch / 序列长度，确认不是被写死的静态 shape ----
    dyn_batch, dyn_src_len = args.batch_size + 1, input_ids.shape[1] + 4
    dyn_tgt_len = max(2, decoder_input_ids.shape[1] - 4)
    src2 = np.random.randint(1, int(config.src_vocab_size), (dyn_batch, dyn_src_len)).astype(np.int64)
    mask2 = np.ones_like(src2)
    tgt2 = np.random.randint(1, int(config.tgt_vocab_size), (dyn_batch, dyn_tgt_len)).astype(np.int64)
    mem2 = enc.run(None, {"input_ids": src2, "attention_mask": mask2})[0]
    out2 = dec.run(None, {"input_ids": tgt2, "encoder_hidden_states": mem2,
                          "encoder_attention_mask": mask2})[0]
    expected_mem = [dyn_batch, dyn_src_len, int(config.d_model)]
    expected_out = [dyn_batch, dyn_tgt_len, int(config.tgt_vocab_size)]
    dynamic_ok = list(mem2.shape) == expected_mem and list(out2.shape) == expected_out
    LOGGER.info("[校验] 动态轴: encoder %s -> %s, decoder %s -> %s  通过=%s",
                (dyn_batch, dyn_src_len), list(mem2.shape), (dyn_batch, dyn_tgt_len),
                list(out2.shape), dynamic_ok)
    report["dynamic_axes"] = {
        "passed": bool(dynamic_ok),
        "memory_shape": list(mem2.shape), "logits_shape": list(out2.shape),
        "expected_memory_shape": expected_mem, "expected_logits_shape": expected_out,
    }
    return report


# ===========================================================================
# 7) HuggingFace 风格的目录产物：config.json / generation_config.json / ...
# ===========================================================================
def build_config_json(output_mode: str, patch_applied: bool) -> dict:
    """对齐 transformers.PretrainedConfig 的字段命名（seq2seq: TransformerConfig / MarianConfig 风格）。"""
    d_model, n_heads, n_layers, d_ff = (int(config.d_model), int(config.n_heads),
                                        int(config.n_layers), int(config.d_ff))
    return {
        "model_type": "transformer",
        "architectures": ["TransformerForConditionalGeneration"],
        "transformers_version": _transformers_version(),
        "is_encoder_decoder": True,
        "d_model": d_model,
        "encoder_layers": n_layers,
        "decoder_layers": n_layers,
        "encoder_attention_heads": n_heads,
        "decoder_attention_heads": n_heads,
        "encoder_ffn_dim": d_ff,
        "decoder_ffn_dim": d_ff,
        "activation_function": "relu",
        "dropout": float(config.dropout),
        "attention_dropout": float(config.dropout),
        "activation_dropout": float(config.dropout),
        "scale_embedding": True,          # Embeddings.forward 里乘了 sqrt(d_model)
        "pre_norm": False,                # 本工程是 Post-LN: x + dropout(sublayer(norm(x)))
        "tie_word_embeddings": False,     # src_embed / tgt_embed / generator 三者不共享权重
        "max_position_embeddings": 5000,  # PositionalEncoding 里 pe 的 max_len
        "vocab_size": int(config.tgt_vocab_size),
        "src_vocab_size": int(config.src_vocab_size),
        "tgt_vocab_size": int(config.tgt_vocab_size),
        "pad_token_id": int(config.padding_idx),
        "unk_token_id": 1,
        "bos_token_id": int(config.bos_idx),
        "eos_token_id": int(config.eos_idx),
        "decoder_start_token_id": int(config.bos_idx),
        # 本工程附加信息（HF 侧会忽略未知字段）
        "output_mode": output_mode,
        "onnx_patch_applied": bool(patch_applied),
    }


def build_generation_config_json() -> dict:
    """对齐 transformers.GenerationConfig 的字段命名。"""
    return {
        "bos_token_id": int(config.bos_idx),
        "eos_token_id": int(config.eos_idx),
        "pad_token_id": int(config.padding_idx),
        "decoder_start_token_id": int(config.bos_idx),
        "max_length": int(config.max_len),
        "min_length": 0,
        "num_beams": int(config.beam_size),
        "num_return_sequences": 1,
        "early_stopping": True,
        "length_penalty": 1.0,
        "do_sample": False,
        "repetition_penalty": 1.0,
    }


def _spm_piece(model_path: str, index: int, default: str) -> str:
    try:
        import sentencepiece as spm
        proc = spm.SentencePieceProcessor()
        proc.Load(model_path)
        return proc.id_to_piece(index)
    except Exception:  # noqa: BLE001
        return default


def build_tokenizer_config_json(src_spm_path: str, tgt_spm_path: str) -> dict:
    """对齐 tokenizer_config.json 的字段命名（source.spm / target.spm 与 HF MarianTokenizer 同构）。"""
    pad, unk, bos, eos = int(config.padding_idx), 1, int(config.bos_idx), int(config.eos_idx)
    return {
        "tokenizer_class": "SentencePieceTokenizer",
        "source_lang": "en",
        "target_lang": "zh",
        # 导出目录里固定叫 source.spm / target.spm（write_artifacts 复制的目标文件名）
        "source_spm": "source.spm",
        "target_spm": "target.spm",
        "pad_token": _spm_piece(src_spm_path, pad, "<pad>"),
        "unk_token": _spm_piece(src_spm_path, unk, "<unk>"),
        "bos_token": _spm_piece(src_spm_path, bos, "<s>"),
        "eos_token": _spm_piece(src_spm_path, eos, "</s>"),
        "pad_token_id": pad,
        "unk_token_id": unk,
        "bos_token_id": bos,
        "eos_token_id": eos,
        "model_max_length": int(config.max_len),
        "add_bos_token": True,
        "add_eos_token": True,
        "clean_up_tokenization_spaces": True,
    }


def build_readme(generation: dict, output_mode: str, ckpt_name: str) -> str:
    """模型卡：front matter 采用 HF 仓库约定（library_name / pipeline_tag / tags）。"""
    max_length = generation["max_length"]
    num_beams = generation["num_beams"]
    early_stopping = str(generation["early_stopping"]).lower()
    length_penalty = generation["length_penalty"]
    d_model = int(config.d_model)
    tgt_vocab = int(config.tgt_vocab_size)
    return f"""---
library_name: transformers
pipeline_tag: translation
tags:
- onnx
- translation
- en-zh
- sentencepiece
---

# Transformer en → zh（ONNX，仅推理）

本目录由 `export_onnx_hf.py` 从 `train_main.py` 训练出的 `{ckpt_name}` 导出，
布局对齐 HuggingFace / Optimum 的 seq2seq ONNX 规范，**只包含推理图**
（`eval()` + `torch.no_grad()` + `TrainingMode.EVAL`，没有 loss / 梯度 / dropout 分支）。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `encoder_model.onnx` | `input_ids(int64, B, S)` + `attention_mask(int64, B, S)` → `last_hidden_state(float32, B, S, {d_model})` |
| `decoder_model.onnx` | `input_ids(int64, B, T)` + `encoder_hidden_states(float32, B, S, {d_model})` + `encoder_attention_mask(int64, B, S)` → `{output_mode}(float32, B, T, {tgt_vocab})` |
| `config.json` | 模型结构（`transformers.PretrainedConfig` 风格） |
| `generation_config.json` | 生成参数：`max_length={max_length}`、`num_beams={num_beams}`、`early_stopping={early_stopping}`、`length_penalty={length_penalty}` |
| `tokenizer_config.json` | sentencepiece 分词器与 special token 映射 |
| `source.spm` / `target.spm` | 英文 / 中文分词模型（目录自包含） |

> 未导出 `decoder_with_past_model.onnx`：本工程的注意力没有实现 KV Cache（每步都重算整段
> self-attention），HF 的 `decoder_with_past` 只有在模型支持 `past_key_values` 时才有意义。

## 用法一：纯 onnxruntime（贪心解码）

```python
import numpy as np
import onnxruntime as ort

enc = ort.InferenceSession("encoder_model.onnx", providers=["CPUExecutionProvider"])
dec = ort.InferenceSession("decoder_model.onnx", providers=["CPUExecutionProvider"])

input_ids = np.array([[2, 105, 408, 2563, 3]], dtype=np.int64)   # <s> ... </s>
attention_mask = np.ones_like(input_ids)
memory = enc.run(None, {{"input_ids": input_ids, "attention_mask": attention_mask}})[0]

cur = np.array([[2]], dtype=np.int64)                            # decoder_start_token_id
for _ in range({max_length}):
    out = dec.run(None, {{"input_ids": cur,
                          "encoder_hidden_states": memory,
                          "encoder_attention_mask": attention_mask}})[0]
    nxt = int(out[0, -1].argmax())
    cur = np.concatenate([cur, [[nxt]]], axis=1)
    if nxt == 3:                                                 # eos
        break
```

## 用法二：本工程提供的 HF 风格封装

```python
from export_onnx_hf import OnnxTranslator

translator = OnnxTranslator.from_pretrained("data/train/exp/weights/onnx_hf")
print(translator.generate(["The cat is sleeping on the sofa."]))
```
"""


def write_artifacts(out_dir: str, ckpt_path: str, args, exported: list, verification: dict,
                    patch_applied: bool, patch_diff: float, src_spm: str | None, tgt_spm: str | None) -> None:
    cfg = build_config_json(args.output, patch_applied)
    gen = build_generation_config_json()
    files = {"config.json": cfg, "generation_config.json": gen}
    if src_spm and tgt_spm:
        files["tokenizer_config.json"] = build_tokenizer_config_json(src_spm, tgt_spm)
    for name, payload in files.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    if src_spm and tgt_spm:
        for src, dst in ((src_spm, "source.spm"), (tgt_spm, "target.spm")):
            shutil.copyfile(src, os.path.join(out_dir, dst))

    readme = build_readme(gen, args.output, os.path.basename(ckpt_path))
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fp:
        fp.write(readme)

    meta = {
        "checkpoint": ckpt_path,
        "onnx_opset": int(args.opset),
        "exporter": args.exporter,
        "output_mode": args.output,
        "exported_at_dtype": "float32",
        "inference_only": True,
        "patch_applied": patch_applied,
        "patch_equivalence_max_abs_diff": patch_diff,
        "files": exported,
        "verification": verification,
        "extra_files": [k for k in files] + ["README.md"],
    }
    with open(os.path.join(out_dir, META_FILE), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)


# ===========================================================================
# 8) 推理封装（HF 风格 API：from_pretrained / generate）
# ===========================================================================
class OnnxTranslator:
    """从导出目录加载 ONNX 模型做推理，接口模仿 transformers 的用法：
        translator = OnnxTranslator.from_pretrained("data/train/exp/weights/onnx_hf")
        translator.generate(["The cat is sleeping on the sofa."])
    """

    def __init__(self, model_dir: str, encoder_session, decoder_session,
                 config_json: dict, generation_config: dict, output_mode: str):
        self.model_dir = model_dir
        self.encoder = encoder_session
        self.decoder = decoder_session
        self.config = config_json
        self.generation_config = generation_config
        self.output_mode = output_mode
        self.pad_token_id = int(config_json.get("pad_token_id", 0))
        self.bos_token_id = int(config_json.get("bos_token_id", 2))
        self.eos_token_id = int(config_json.get("eos_token_id", 3))
        self.decoder_start_token_id = int(config_json.get("decoder_start_token_id",
                                                          self.bos_token_id))
        self.max_length = int(generation_config.get("max_length", 60))
        self.num_beams = int(generation_config.get("num_beams", 1))
        self.length_penalty = float(generation_config.get("length_penalty", 1.0))
        self.early_stopping = bool(generation_config.get("early_stopping", True))
        self.source_sp = None
        self.target_sp = None

    # ---------------- 加载 ----------------
    @classmethod
    def from_pretrained(cls, model_dir: str, coreml: bool = False) -> "OnnxTranslator":
        import onnxruntime as ort

        model_dir = os.path.abspath(resolve_path(model_dir))
        required = [ENCODER_FILE, DECODER_FILE, "config.json", "generation_config.json"]
        missing = [f for f in required if not os.path.isfile(os.path.join(model_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"{model_dir} 缺少 {missing}，请先执行: python export_onnx_hf.py --out-dir {model_dir}")

        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fp:
            config_json = json.load(fp)
        with open(os.path.join(model_dir, "generation_config.json"), encoding="utf-8") as fp:
            generation_config = json.load(fp)
        output_mode = config_json.get("output_mode", "logits")

        providers = ["CPUExecutionProvider"]
        if coreml and "CoreMLExecutionProvider" in ort.get_available_providers():
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        enc = ort.InferenceSession(os.path.join(model_dir, ENCODER_FILE), providers=providers)
        dec = ort.InferenceSession(os.path.join(model_dir, DECODER_FILE), providers=providers)

        obj = cls(model_dir, enc, dec, config_json, generation_config, output_mode)
        obj._load_tokenizers()
        LOGGER.info("已加载 ONNX 模型: %s（output_mode=%s, providers=%s）",
                    model_dir, output_mode, providers)
        return obj

    def _load_tokenizers(self) -> None:
        import sentencepiece as spm

        tok_cfg_path = os.path.join(self.model_dir, "tokenizer_config.json")
        defaults = {"source_spm": "source.spm", "target_spm": "target.spm"}
        if os.path.isfile(tok_cfg_path):
            with open(tok_cfg_path, encoding="utf-8") as fp:
                tok_cfg = json.load(fp)
            self.pad_token_id = int(tok_cfg.get("pad_token_id", self.pad_token_id))
            self.bos_token_id = int(tok_cfg.get("bos_token_id", self.bos_token_id))
            self.eos_token_id = int(tok_cfg.get("eos_token_id", self.eos_token_id))
            defaults.update({k: tok_cfg[k] for k in ("source_spm", "target_spm") if k in tok_cfg})

        for attr, key in (("source_sp", "source_spm"), ("target_sp", "target_spm")):
            path = os.path.join(self.model_dir, defaults[key])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"缺少分词模型 {path}（可加 --no-tokenizer 后手动拷贝 .spm）")
            proc = spm.SentencePieceProcessor()
            proc.Load(path)
            setattr(self, attr, proc)

    # ---------------- 文本 <-> id ----------------
    def _encode(self, texts) -> "tuple":
        import numpy as np

        sequences = [[self.bos_token_id] + self.source_sp.EncodeAsIds(t) + [self.eos_token_id]
                     for t in texts]
        length = max(len(s) for s in sequences)
        input_ids = np.full((len(sequences), length), self.pad_token_id, dtype=np.int64)
        for i, ids in enumerate(sequences):
            input_ids[i, :len(ids)] = ids
        attention_mask = (input_ids != self.pad_token_id).astype(np.int64)
        return input_ids, attention_mask

    def _decode_text(self, ids) -> str:
        cleaned = []
        for tid in ids:
            if tid == self.eos_token_id:
                break
            if tid in (self.pad_token_id, self.bos_token_id):
                continue
            cleaned.append(int(tid))
        return self.target_sp.decode_ids(cleaned)

    # ---------------- 推理 ----------------
    def encode(self, texts):
        """等价 encoder_model.onnx 的前向：英文文本 -> encoder_hidden_states。"""
        input_ids, attention_mask = self._encode(texts)
        memory = self.encoder.run(None, {"input_ids": input_ids,
                                         "attention_mask": attention_mask})[0]
        return memory, attention_mask

    def _last_log_probs(self, logits):
        import numpy as np

        if self.output_mode == "log_probs":
            return logits
        shifted = logits - logits.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    def _decoder_step(self, input_ids, memory, attention_mask):
        logits = self.decoder.run(None, {
            "input_ids": input_ids,
            "encoder_hidden_states": memory,
            "encoder_attention_mask": attention_mask,
        })[0]
        return self._last_log_probs(logits[:, -1, :])

    def _greedy(self, memory, attention_mask, max_length: int) -> list:
        import numpy as np

        generated = [self.decoder_start_token_id]
        for _ in range(max_length - 1):
            cur = np.array([generated], dtype=np.int64)
            nxt = int(self._decoder_step(cur, memory, attention_mask)[0].argmax())
            generated.append(nxt)
            if nxt == self.eos_token_id:
                break
        return generated

    def _beam_search(self, memory, attention_mask, max_length: int, num_beams: int) -> list:
        import numpy as np

        beams = [([self.decoder_start_token_id], 0.0)]
        finished = []
        for _ in range(max_length - 1):
            if not beams:
                break
            longest = max(len(ids) for ids, _ in beams)
            cur = np.full((len(beams), longest), self.pad_token_id, dtype=np.int64)
            for i, (ids, _) in enumerate(beams):
                cur[i, :len(ids)] = ids
            mem = np.repeat(memory, len(beams), axis=0)
            mask = np.repeat(attention_mask, len(beams), axis=0)
            log_probs = self._decoder_step(cur, mem, mask)

            candidates = []
            for i, (ids, score) in enumerate(beams):
                top = np.argsort(-log_probs[i])[:num_beams]
                for tid in top:
                    candidates.append((ids + [int(tid)], score + float(log_probs[i][tid])))
            candidates.sort(key=lambda item: item[1], reverse=True)

            beams, finished = [], finished
            for ids, score in candidates:
                if ids[-1] == self.eos_token_id:
                    finished.append((ids, score))
                elif len(beams) < num_beams:
                    beams.append((ids, score))
                if len(beams) >= num_beams and (not self.early_stopping or len(finished) >= num_beams):
                    break
            if self.early_stopping and len(finished) >= num_beams:
                break

        pool = finished or beams
        if not pool:
            return [self.decoder_start_token_id]
        pool.sort(key=lambda item: item[1] / max(1, len(item[0]) - 1) ** self.length_penalty,
                  reverse=True)
        return pool[0][0]

    def generate(self, texts, max_length: int = None, num_beams: int = None,
                 show_steps: bool = False) -> list:
        """HF 风格 generate：输入英文句子（str 或 list），返回中文译文列表。"""
        if isinstance(texts, str):
            texts = [texts]
        max_length = int(max_length or self.max_length)
        num_beams = int(num_beams if num_beams is not None else self.num_beams)

        outputs = []
        for text in texts:
            memory, attention_mask = self.encode([text])
            if num_beams and num_beams > 1:
                ids = self._beam_search(memory, attention_mask, max_length, num_beams)
            else:
                ids = self._greedy(memory, attention_mask, max_length)
            if show_steps:
                LOGGER.info("  token ids(%d): %s", len(ids), ids)
            outputs.append(self._decode_text(ids))
        return outputs


# ===========================================================================
# 9) 命令行
# ===========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="export_onnx_hf.py",
        description="按 HuggingFace/Optimum 规范把 train_main.py 的 .pth 权重导出成 ONNX（只含推理图）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", default=config.translate_model_path,
                        help="PyTorch 权重路径，默认取 config.translate_model_path")
    parser.add_argument("--out-dir", default=None, help="导出目录，默认 <ckpt目录>/onnx_hf")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset 版本，<=0 表示交给 PyTorch 自动选择")
    parser.add_argument("--exporter", choices=["auto", "dynamo", "legacy"], default="auto",
                        help="导出后端：auto / dynamo(torch.export) / legacy(TorchScript)")
    parser.add_argument("--patch", choices=["auto", "always", "never"], default="auto",
                        help="ONNX 友好化补丁：auto=直接导出失败时才打，always=总是打，never=从不打")
    parser.add_argument("--output", choices=["logits", "log_probs"], default="logits",
                        help="decoder 输出：HF 约定的 logits，或原 Generator 的 log_probs")
    parser.add_argument("--batch-size", type=int, default=2, help="导出用 dummy 输入的 batch size")
    parser.add_argument("--src-len", type=int, default=16, help="导出用 dummy 输入的源句长度")
    parser.add_argument("--tgt-len", type=int, default=16, help="导出用 dummy 输入的目标句长度")
    parser.add_argument("--no-tokenizer", dest="tokenizer", action="store_false",
                        help="不复制 source.spm / target.spm、不写 tokenizer_config.json")
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="跳过导出后校验")
    parser.add_argument("--atol", type=float, default=1e-3, help="数值校验的绝对容差")
    parser.add_argument("--rtol", type=float, default=1e-3, help="数值校验的相对容差")
    parser.add_argument("--seed", type=int, default=0, help="dummy 输入随机种子")
    parser.add_argument("--clean", action="store_true", help="导出前清理目标目录里的旧产物")
    parser.add_argument("--coreml", action="store_true", help="校验时使用 CoreML EP（更快，但可能有数值差异）")
    parser.add_argument("--demo", action="store_true", help="导出后加载 ONNX 目录，跑一次真实翻译")
    parser.add_argument("--text", action="append", default=None, help="demo 要翻译的英文句子，可重复指定")
    parser.add_argument("--from-dev", type=int, default=3, help="未指定 --text 时，demo 从 dev.json 取前 N 句")
    parser.add_argument("--max-len", type=int, default=config.max_len, help="demo 解码最大长度")
    parser.add_argument("--beam-size", type=int, default=config.beam_size, help="demo 的 beam size，1 表示贪心")
    return parser.parse_args(argv)


def _load_demo_pairs(args) -> list:
    """demo 用的 (英文, 参考中文) 句对；用 --text 时参考译文为空串。"""
    if args.text:
        return [(t, "") for t in args.text]
    path = os.path.join(SCRIPT_DIR, config.dev_data_path.lstrip("./"))
    if not os.path.isfile(path):
        LOGGER.warning("找不到 %s，改用内置示例句", path)
        samples = [("I love you.", "我爱你。"),
                   ("Renewing the South Korean Miracle", "再造韩国奇迹")]
        return samples[: max(1, args.from_dev)]
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    return [(row[0], row[1]) for row in data[: max(1, args.from_dev)]]


def torch_reference_greedy(model, source_sp, text, max_len, padding_idx, bos_idx, eos_idx) -> list:
    """用 PyTorch 原模型（未打补丁）做贪心解码，作为 ONNX 的对照基准，返回 token id 列表。"""
    with torch.no_grad():
        src = torch.LongTensor([[bos_idx] + source_sp.EncodeAsIds(text) + [eos_idx]])
        src_mask = (src != padding_idx).unsqueeze(-2)
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), bos_idx, dtype=torch.long)
        for _ in range(max_len - 1):
            length = ys.size(1)
            tgt_mask = (ys != padding_idx).unsqueeze(-2) & torch.tril(
                torch.ones(length, length, dtype=torch.bool))
            out = model.decode(memory, src_mask, ys, tgt_mask)
            nxt = int(model.generator(out[:, -1]).argmax(dim=-1).item())
            ys = torch.cat([ys, torch.full((1, 1), nxt, dtype=torch.long)], dim=1)
            if nxt == eos_idx:
                break
    return ys[0].tolist()


def _cut_at_eos(ids, eos_idx) -> list:
    """截断到 EOS（含 EOS），忽略其后的 padding，方便逐 token 比较。"""
    return list(ids[: ids.index(eos_idx) + 1]) if eos_idx in ids else list(ids)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    ckpt_path = resolve_path(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "onnx_hf")
    out_dir = os.path.abspath(resolve_path(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    if args.clean:
        for name in GENERATED_FILES:
            target = os.path.join(out_dir, name)
            if os.path.isfile(target):
                os.remove(target)
        LOGGER.info("已清理目标目录旧产物: %s", out_dir)

    # ---- 构建模型 & 加载权重（eval + no_grad：只做推理）----
    model = build_and_load_model(ckpt_path, device="cpu")
    padding_idx = int(config.padding_idx)

    # ---- dummy 输入 & 导出配置 ----
    onnx_config = OnnxSeq2SeqConfig(args, args.output, padding_idx)
    input_ids, attention_mask, decoder_input_ids = onnx_config.dummy_inputs(
        max(1, args.batch_size), max(1, args.src_len), max(2, args.tgt_len))

    encoder_module = HFEncoderModel(model)
    decoder_module = HFDecoderModel(model, padding_idx, args.output)
    encoder_module.eval()
    decoder_module.eval()

    dynamo_supported = "dynamo" in inspect.signature(torch.onnx.export).parameters
    if not dynamo_supported:
        LOGGER.warning("当前 torch.onnx.export 不支持 dynamo 参数，将只使用 legacy 导出器")

    def _apply_patch() -> float:
        """打上 ONNX 友好化补丁，并返回补丁前后（decoder 输出）的最大差异。"""
        with torch.no_grad():
            before = decoder_module(decoder_input_ids, encoder_module(input_ids, attention_mask),
                                    attention_mask)
        patch_model_for_onnx()
        with torch.no_grad():
            after = decoder_module(decoder_input_ids, encoder_module(input_ids, attention_mask),
                                   attention_mask)
        diff = float((after - before).abs().max())
        LOGGER.info("补丁等价性检查: max_abs_diff=%.3e（越接近 0 说明补丁越无损）", diff)
        return diff

    # ---- 导出：默认先用"原封不动"的模型导出，只有失败时才启用补丁重试 ----
    patch_applied = args.patch == "always"
    patch_diff = _apply_patch() if patch_applied else 0.0
    try:
        exported = export_models(encoder_module, decoder_module, input_ids, attention_mask,
                                 decoder_input_ids, onnx_config, out_dir, args, dynamo_supported)
    except RuntimeError as exc:
        if patch_applied or args.patch == "never":
            raise
        LOGGER.warning("未打补丁直接导出失败，改用 ONNX 友好化补丁后重试: %s", str(exc)[:300])
        patch_diff = _apply_patch()
        patch_applied = True
        exported = export_models(encoder_module, decoder_module, input_ids, attention_mask,
                                 decoder_input_ids, onnx_config, out_dir, args, dynamo_supported)

    # ---- 校验 ----
    verification = {}
    if args.verify:
        verification = verify_onnx(os.path.join(out_dir, ENCODER_FILE),
                                   os.path.join(out_dir, DECODER_FILE),
                                   encoder_module, decoder_module,
                                   input_ids, attention_mask, decoder_input_ids, args)

    # ---- HF 风格的目录产物 ----
    src_spm = tgt_spm = None
    if args.tokenizer:
        src_spm = resolve_path("tokenizer/eng.model")
        tgt_spm = resolve_path("tokenizer/chn.model")
        if not (os.path.isfile(src_spm) and os.path.isfile(tgt_spm)):
            LOGGER.warning("找不到 tokenizer/eng.model 或 tokenizer/chn.model，跳过分词器产物")
            src_spm = tgt_spm = None
    write_artifacts(out_dir, ckpt_path, args, exported, verification,
                    patch_applied, patch_diff, src_spm, tgt_spm)

    LOGGER.info("导出完成，输出目录: %s", out_dir)
    LOGGER.info("目录结构（HuggingFace 风格，仅推理）:")
    for name in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, name)) / 1024 / 1024
        LOGGER.info("  - %-24s %8.2f MB", name, size)

    # ---- 可选：HF 风格推理 demo（ONNX 解码 + 与 PyTorch 原模型逐 token 对比）----
    if args.demo:
        translator = OnnxTranslator.from_pretrained(out_dir, coreml=args.coreml)
        pairs = _load_demo_pairs(args)
        translations = translator.generate([en for en, _ in pairs], max_length=args.max_len,
                                           num_beams=args.beam_size)
        torch_model = build_and_load_model(ckpt_path, device="cpu")
        bos_idx, eos_idx = int(config.bos_idx), int(config.eos_idx)

        all_matched = True
        for (en, zh_ref), zh in zip(pairs, translations):
            memory, attention_mask = translator.encode([en])
            onnx_greedy_ids = translator._greedy(memory, attention_mask, args.max_len)
            torch_ids = torch_reference_greedy(torch_model, translator.source_sp, en,
                                               args.max_len, padding_idx, bos_idx, eos_idx)
            matched = _cut_at_eos(onnx_greedy_ids, eos_idx) == _cut_at_eos(torch_ids, eos_idx)
            all_matched = all_matched and matched

            print("\n" + "-" * 100)
            print(f"EN              : {en}")
            if zh_ref:
                print(f"参考译文         : {zh_ref}")
            print(f"ONNX 贪心        : {translator._decode_text(onnx_greedy_ids)}")
            print(f"ONNX {args.beam_size}-beam       : {zh}")
            print(f"PyTorch 贪心     : {translator._decode_text(torch_ids)}")
            print(f"token 级一致     : {matched}  (ONNX {len(onnx_greedy_ids)} tokens, "
                  f"PyTorch {len(torch_ids)} tokens)")
        print("\n" + "=" * 100)
        print("ONNX 与 PyTorch 逐 token 完全一致 ✅" if all_matched else "存在不一致 ❌")
    return 0


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
