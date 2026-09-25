#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 train_main.py 训练出来的 PyTorch 权重（.pth）导出成 ONNX 模型。

默认导出 3 个文件（放在 <ckpt 所在目录>/onnx 下）：

    transformer_full.onnx     输入: src(int64, B,S), tgt(int64, B,T)            输出: log_probs(B,T,V)
    transformer_encoder.onnx  输入: src(int64, B,S)                             输出: memory(B,S,D)
    transformer_decoder.onnx  输入: tgt(int64, B,T), memory(B,S,D),
                                     src_mask(bool, B,1,S)                      输出: log_probs(B,T,V)

说明：
    1. 原模型的 forward 只返回 decoder 的 hidden，这里包一层把 generator 也导出，
       所以 ONNX 的输出是 log_softmax 之后的概率（--output logits 可改为输出原始 logits）。
    2. 原先 mask 是在 data_loader 里算好再喂给模型的，这里把 mask 的构造挪进图内，
       规则与原实现完全一致：
           src_mask = (src != padding_idx).unsqueeze(-2)                -> (B, 1, S)
           tgt_mask = (tgt != padding_idx).unsqueeze(-2) & tril(ones(T,T)) -> (B, T, T)
       拆分模型里 decoder 无法由 memory 反推 padding，因此 src_mask 作为输入传入。
    3. 默认用“原封不动”的模型直接导出（torch>=2.9 的新导出器可直接处理 x.std()）。
       只有在导出失败时（--patch auto），才会自动打上数学等价的 ONNX 友好化补丁并重试，
       同时打印补丁前后的输出误差；也可以用 --patch always/never 强制开启或关闭：
           - PositionalEncoding.forward 里的 torch.autograd.Variable 包装（已废弃 API）
           - LayerNorm.forward 里的 x.std()（无偏标准差，旧版 TorchScript 导出器支持不稳定）

用法：
    # 在 transformers_learning 目录下执行
    python export_onnx.py                                   # 用 config.translate_model_path
    python export_onnx.py --ckpt data/train/exp/weights/best_bleu_26.30.pth
    python export_onnx.py --ckpt xxx.pth --output logits --no-split --out-dir ./onnx_out

    导出后端需要额外依赖（缺哪个装哪个）：
        新导出器（默认，torch>=2.9 的默认后端）: pip install onnx onnxscript
        旧导出器（--exporter legacy）           : pip install onnx
        数值校验（可选，安装后自动开启）        : pip install onnxruntime

ONNXRuntime 推理示例（贪心解码，配合 tokenizer/tokenize.py 使用）：

    import numpy as np
    import onnxruntime as ort

    enc = ort.InferenceSession(".../transformer_encoder.onnx", providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(".../transformer_decoder.onnx", providers=["CPUExecutionProvider"])

    PAD, BOS, EOS, MAX_LEN = 0, 2, 3, 60
    src = np.array([[BOS] + src_ids], dtype=np.int64)          # (1, S)
    src_mask = (src != PAD)[:, None, :]                        # (1, 1, S)

    memory = enc.run(None, {"src": src})[0]                    # (1, S, D)
    tgt = np.array([[BOS]], dtype=np.int64)                    # (1, 1)
    for _ in range(MAX_LEN):
        log_probs = dec.run(None, {"tgt": tgt, "memory": memory, "src_mask": src_mask})[0]
        nxt = int(log_probs[0, -1].argmax())
        tgt = np.concatenate([tgt, [[nxt]]], axis=1)
        if nxt == EOS:
            break
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import logging
import os
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
LOGGER = logging.getLogger("export_onnx")


# ---------------------------------------------------------------------------
# 1. mask 构造（与训练/推理时 data_loader 中的规则保持一致）
# ---------------------------------------------------------------------------
def make_src_mask(src: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """padding mask: (B, S) -> (B, 1, S)，广播到 (B, h, S, S)"""
    return (src != padding_idx).unsqueeze(-2)


def make_tgt_mask(tgt: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """decoder 的 self-attention mask: padding mask & 下三角（禁止看到未来） -> (B, T, T)"""
    tgt_len = tgt.size(1)
    causal = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt.device))
    return (tgt != padding_idx).unsqueeze(-2) & causal


# ---------------------------------------------------------------------------
# 2. ONNX 友好化补丁（数学等价，仅在导出脚本里生效）
# ---------------------------------------------------------------------------
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
        std = torch.sqrt(var)
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2

    tf_model.PositionalEncoding.forward = positional_encoding_forward
    tf_model.LayerNorm.forward = layer_norm_forward
    LOGGER.info("已应用 ONNX 友好化补丁: PositionalEncoding.forward / LayerNorm.forward")


# ---------------------------------------------------------------------------
# 3. 导出用的包装模块（补上 generator + 图内生成 mask）
# ---------------------------------------------------------------------------
class _GeneratorMixin:
    """统一处理 generator 的两种输出形式。"""

    def _init_generator(self, model, generator_mode: str):
        self.model = model
        self.generator_mode = generator_mode

    def _generate(self, decoder_out: torch.Tensor) -> torch.Tensor:
        if self.generator_mode == "logits":
            # Generator.forward = log_softmax(proj(x))，这里只要 proj 的输出
            return self.model.generator.proj(decoder_out)
        return self.model.generator(decoder_out)


class ONNXTransformerFull(nn.Module, _GeneratorMixin):
    """整模型: (src, tgt) -> log_probs，等价于 train_main 里的 forward + generator"""

    def __init__(self, model, padding_idx: int, generator_mode: str = "log_probs"):
        super().__init__()
        self._init_generator(model, generator_mode)
        self.padding_idx = int(padding_idx)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = make_src_mask(src, self.padding_idx)
        tgt_mask = make_tgt_mask(tgt, self.padding_idx)
        out = self.model(src, tgt, src_mask, tgt_mask)
        return self._generate(out)


class ONNXTransformerEncoder(nn.Module):
    """编码器: src -> memory，用于自回归解码时只编码一次"""

    def __init__(self, model, padding_idx: int):
        super().__init__()
        self.model = model
        self.padding_idx = int(padding_idx)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.model.encode(src, make_src_mask(src, self.padding_idx))


class ONNXTransformerDecoder(nn.Module, _GeneratorMixin):
    """解码器: (tgt, memory, src_mask) -> log_probs"""

    def __init__(self, model, padding_idx: int, generator_mode: str = "log_probs"):
        super().__init__()
        self._init_generator(model, generator_mode)
        self.padding_idx = int(padding_idx)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        tgt_mask = make_tgt_mask(tgt, self.padding_idx)
        out = self.model.decode(memory, src_mask, tgt, tgt_mask)
        return self._generate(out)


# ---------------------------------------------------------------------------
# 4. 权重加载
# ---------------------------------------------------------------------------
def resolve_path(path: str) -> str:
    """相对路径优先按“当前工作目录”解析，再按“脚本所在目录”解析。"""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(SCRIPT_DIR, path)
    return candidate if os.path.exists(candidate) else path


def load_state_dict(ckpt_path: str, map_location: str = "cpu") -> dict:
    """加载 .pth；兼容 {'state_dict': ...} 包装与 DataParallel 的 'module.' 前缀。"""
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except Exception as exc:  # 老版本/带自定义对象的 checkpoint 回退到 weights_only=False
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
    LOGGER.info(
        "构建模型: d_model=%d, n_heads=%d, n_layers=%d, d_ff=%d, dropout=%.2f, vocab=%d/%d",
        config.d_model, config.n_heads, config.n_layers, config.d_ff, config.dropout,
        config.src_vocab_size, config.tgt_vocab_size,
    )
    model = make_model(
        config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
        config.d_model, config.d_ff, config.n_heads, config.dropout,
    )
    # 训练可能在 mps/cuda 上完成，ONNX 导出必须在 CPU 上做
    model.to(device)

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
    model.eval()
    LOGGER.info("权重加载完成: %s (共 %d 个张量)", ckpt_path, len(state_dict))
    return model


# ---------------------------------------------------------------------------
# 5. 导出 & 校验
# ---------------------------------------------------------------------------
def _exporter_candidates(preferred: str):
    """返回尝试顺序；dynamo=True 表示新的 torch.export 导出器。"""
    if preferred == "dynamo":
        return [True, False]
    if preferred == "legacy":
        return [False, True]
    has_onnxscript = importlib.util.find_spec("onnxscript") is not None
    return [True, False] if has_onnxscript else [False, True]


def _filter_kwargs(func, kwargs: dict) -> dict:
    accepted = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in accepted}


def export_onnx(module, args, path, input_names, output_names, dynamic_axes,
                opset: int, exporter: str = "auto", dynamo_supported: bool = True):
    """按候选后端依次尝试导出，返回真正生效的后端名。"""
    kwargs = {
        "input_names": list(input_names),
        "output_names": list(output_names),
        "external_data": False,     # 模型 <2GB，强制单文件，避免生成 .onnx.data
        "verbose": False,
    }
    if opset and opset > 0:
        kwargs["opset_version"] = int(opset)
    if dynamic_axes:
        kwargs["dynamic_axes"] = dynamic_axes
    kwargs = _filter_kwargs(torch.onnx.export, kwargs)

    last_error = None
    for dynamo in _exporter_candidates(exporter):
        if dynamo and not dynamo_supported:
            continue
        call_kwargs = dict(kwargs)
        call_kwargs["dynamo"] = dynamo
        if not dynamo:  # 旧导出器专有参数
            call_kwargs.update({"export_params": True, "do_constant_folding": True})
            call_kwargs = _filter_kwargs(torch.onnx.export, call_kwargs)

        backend = "dynamo(torch.export)" if dynamo else "legacy(TorchScript)"
        try:
            with torch.no_grad():
                torch.onnx.export(module, args, path, **call_kwargs)
            LOGGER.info("导出成功 [%s] -> %s (%.1f MB)", backend, path, os.path.getsize(path) / 1024 / 1024)
            return backend
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.warning("使用 %s 导出 %s 失败: %s: %s", backend, os.path.basename(path),
                           type(exc).__name__, str(exc)[:400])

    raise RuntimeError(
        f"ONNX 导出失败: {os.path.basename(path)}，最后一个错误: {last_error}\n"
        "请先安装依赖: pip install onnx onnxscript   # 或缺 onnxscript 时用 --exporter legacy"
    )


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor


def verify_with_onnxruntime(onnx_path: str, module: nn.Module, inputs, atol: float, rtol: float):
    """用 onnxruntime 跑一遍 ONNX，与 PyTorch 输出对比；未安装 onnxruntime 时跳过。"""
    if importlib.util.find_spec("onnxruntime") is None:
        LOGGER.warning("未安装 onnxruntime，跳过数值校验（pip install onnxruntime 后可自动开启）")
        return None

    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_defs = session.get_inputs()
    if len(input_defs) != len(inputs):
        raise RuntimeError(f"{onnx_path} 输入数量不匹配: onnx={len(input_defs)}, torch={len(inputs)}")

    feed = {d.name: _to_numpy(t) for d, t in zip(input_defs, inputs)}
    onnx_outputs = session.run(None, feed)
    with torch.no_grad():
        torch_outputs = module(*inputs)
    if not isinstance(torch_outputs, (tuple, list)):
        torch_outputs = [torch_outputs]

    results = []
    for idx, (onnx_out, torch_out) in enumerate(zip(onnx_outputs, torch_outputs)):
        ref = _to_numpy(torch_out)
        diff = np.abs(onnx_out - ref)
        max_abs = float(diff.max()) if diff.size else 0.0
        max_rel = float((diff / np.maximum(np.abs(ref), 1e-6)).max()) if diff.size else 0.0
        passed = bool(np.allclose(onnx_out, ref, atol=atol, rtol=rtol))
        results.append({"index": idx, "shape": list(ref.shape), "max_abs_diff": max_abs,
                        "max_rel_diff": max_rel, "passed": passed})
        LOGGER.info("[校验] %s 输出%d shape=%s max_abs_diff=%.3e max_rel_diff=%.3e 通过=%s",
                    os.path.basename(onnx_path), idx, list(ref.shape), max_abs, max_rel, passed)
    return results


def export_models(full_module, encoder_module, decoder_module, src, tgt, src_mask,
                  out_dir: str, args, dynamo_supported: bool):
    """导出整模型 + （可选）encoder/decoder 拆分模型，返回 (exported, split_info)。"""
    exported = []
    split_info = None

    full_path = os.path.join(out_dir, "transformer_full.onnx")
    backend = export_onnx(
        full_module, (src, tgt), full_path,
        input_names=["src", "tgt"], output_names=["log_probs"],
        dynamic_axes={"src": {0: "batch", 1: "src_len"},
                      "tgt": {0: "batch", 1: "tgt_len"},
                      "log_probs": {0: "batch", 1: "tgt_len"}},
        opset=args.opset, exporter=args.exporter, dynamo_supported=dynamo_supported,
    )
    exported.append({"file": full_path, "backend": backend,
                     "inputs": {"src": "int64 (batch, src_len)", "tgt": "int64 (batch, tgt_len)"},
                     "outputs": {"log_probs": "float32 (batch, tgt_len, tgt_vocab)"}})

    if encoder_module is not None and decoder_module is not None:
        enc_path = os.path.join(out_dir, "transformer_encoder.onnx")
        enc_backend = export_onnx(
            encoder_module, (src,), enc_path,
            input_names=["src"], output_names=["memory"],
            dynamic_axes={"src": {0: "batch", 1: "src_len"},
                          "memory": {0: "batch", 1: "src_len"}},
            opset=args.opset, exporter=args.exporter, dynamo_supported=dynamo_supported,
        )
        exported.append({"file": enc_path, "backend": enc_backend,
                         "inputs": {"src": "int64 (batch, src_len)"},
                         "outputs": {"memory": "float32 (batch, src_len, d_model)"}})

        with torch.no_grad():
            memory = encoder_module(src)
        dec_path = os.path.join(out_dir, "transformer_decoder.onnx")
        dec_backend = export_onnx(
            decoder_module, (tgt, memory, src_mask), dec_path,
            input_names=["tgt", "memory", "src_mask"], output_names=["log_probs"],
            dynamic_axes={"tgt": {0: "batch", 1: "tgt_len"},
                          "memory": {0: "batch", 1: "src_len"},
                          "src_mask": {0: "batch", 2: "src_len"},
                          "log_probs": {0: "batch", 1: "tgt_len"}},
            opset=args.opset, exporter=args.exporter, dynamo_supported=dynamo_supported,
        )
        exported.append({"file": dec_path, "backend": dec_backend,
                         "inputs": {"tgt": "int64 (batch, tgt_len)",
                                    "memory": "float32 (batch, src_len, d_model)",
                                    "src_mask": "bool (batch, 1, src_len)"},
                         "outputs": {"log_probs": "float32 (batch, tgt_len, tgt_vocab)"}})
        split_info = {"encoder": enc_path, "decoder": dec_path}

    return exported, split_info


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="export_onnx.py",
        description="把 train_main.py 训练出的 .pth 权重导出成 ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", default=config.translate_model_path,
                        help="PyTorch 权重路径，默认取 config.translate_model_path")
    parser.add_argument("--out-dir", default=None, help="ONNX 输出目录，默认 <ckpt目录>/onnx")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset 版本，<=0 表示交给 PyTorch 自动选择")
    parser.add_argument("--exporter", choices=["auto", "dynamo", "legacy"], default="auto",
                        help="导出后端：auto/dynamo(torch.export)/legacy(TorchScript)")
    parser.add_argument("--patch", choices=["auto", "always", "never"], default="auto",
                        help="ONNX 友好化补丁：auto=直接导出失败时才打，always=总是打，never=从不打")
    parser.add_argument("--output", choices=["log_probs", "logits"], default="log_probs",
                        help="模型输出 log_softmax 概率还是原始 logits")
    parser.add_argument("--batch-size", type=int, default=2, help="导出用 dummy 输入的 batch size")
    parser.add_argument("--src-len", type=int, default=16, help="导出用 dummy 输入的源句长度")
    parser.add_argument("--tgt-len", type=int, default=16, help="导出用 dummy 输入的目标句长度")
    parser.add_argument("--no-split", dest="split", action="store_false",
                        help="只导出整模型，不额外导出 encoder/decoder 拆分模型")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="跳过 onnxruntime 数值校验")
    parser.add_argument("--atol", type=float, default=1e-3, help="数值校验的绝对容差")
    parser.add_argument("--rtol", type=float, default=1e-3, help="数值校验的相对容差")
    parser.add_argument("--seed", type=int, default=0, help="dummy 输入随机种子")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    ckpt_path = resolve_path(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "onnx")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 构建模型 & 加载权重 ----
    model = build_and_load_model(ckpt_path, device=config.device)
    padding_idx = int(config.padding_idx)

    # ---- dummy 输入 ----
    batch_size = max(1, args.batch_size)
    src_len = max(1, args.src_len)
    tgt_len = max(2, args.tgt_len)  # 至少 2，保证 causal mask 有意义
    vocab_src = int(config.src_vocab_size)
    vocab_tgt = int(config.tgt_vocab_size)
    # 用 [1, vocab) 之间的 id，避免整句都是 padding
    src = torch.randint(1, vocab_src, (batch_size, src_len), dtype=torch.long)
    tgt = torch.randint(1, vocab_tgt, (batch_size, tgt_len), dtype=torch.long)
    src_mask = make_src_mask(src, padding_idx)

    # ---- 组装包装模块（补上 generator，并把 mask 生成放进图里）----
    full_module = ONNXTransformerFull(model, padding_idx, args.output)
    if args.split:
        encoder_module = ONNXTransformerEncoder(model, padding_idx)
        decoder_module = ONNXTransformerDecoder(model, padding_idx, args.output)
    else:
        encoder_module = decoder_module = None

    dynamo_supported = "dynamo" in inspect.signature(torch.onnx.export).parameters
    if not dynamo_supported:
        LOGGER.warning("当前 torch.onnx.export 不支持 dynamo 参数，将只使用 legacy 导出器")

    def _apply_patch() -> float:
        """打上 ONNX 友好化补丁，并返回补丁前后输出（原模型语义）的最大差异。"""
        with torch.no_grad():
            before = full_module(src, tgt)
        patch_model_for_onnx()
        with torch.no_grad():
            diff = (full_module(src, tgt) - before).abs().max().item()
        LOGGER.info("补丁等价性检查: max_abs_diff=%.3e（越接近 0 说明补丁越无损）", diff)
        return diff

    # ---- 导出：默认先用“原封不动”的模型导出，只有失败时才启用补丁重试 ----
    patch_applied = args.patch == "always"
    patch_diff = _apply_patch() if patch_applied else 0.0
    export_kwargs = {"src": src, "tgt": tgt, "src_mask": src_mask, "out_dir": out_dir, "args": args}
    try:
        exported, split_info = export_models(full_module, encoder_module, decoder_module,
                                             dynamo_supported=dynamo_supported, **export_kwargs)
    except RuntimeError as exc:
        if patch_applied or args.patch == "never":
            raise
        LOGGER.warning("未打补丁直接导出失败，改用 ONNX 友好化补丁后重试: %s", str(exc)[:300])
        patch_diff = _apply_patch()
        patch_applied = True
        exported, split_info = export_models(full_module, encoder_module, decoder_module,
                                             dynamo_supported=dynamo_supported, **export_kwargs)

    # ---- 数值校验（onnxruntime 实跑，与 PyTorch 逐值对比）----
    verification = {}
    if args.verify and exported:
        verification["full"] = verify_with_onnxruntime(
            exported[0]["file"], full_module, (src, tgt), args.atol, args.rtol)
        if split_info:
            with torch.no_grad():
                memory = encoder_module(src)
            verification["encoder"] = verify_with_onnxruntime(
                split_info["encoder"], encoder_module, (src,), args.atol, args.rtol)
            verification["decoder"] = verify_with_onnxruntime(
                split_info["decoder"], decoder_module, (tgt, memory, src_mask), args.atol, args.rtol)

    # ---- 记录元信息，方便推理侧对齐 tokenizer / 特殊符号 ----
    meta = {
        "checkpoint": ckpt_path,
        "opset": args.opset,
        "output_mode": args.output,
        "model": {
            "src_vocab_size": vocab_src, "tgt_vocab_size": vocab_tgt,
            "d_model": config.d_model, "n_heads": config.n_heads, "n_layers": config.n_layers,
            "d_ff": config.d_ff, "dropout": config.dropout,
            "padding_idx": padding_idx, "bos_idx": int(config.bos_idx), "eos_idx": int(config.eos_idx),
            "max_len": int(config.max_len), "beam_size": int(config.beam_size),
        },
        "tokenizer": {
            "chinese_model": "tokenizer/chn.model", "chinese_vocab": "tokenizer/chn.vocab",
            "english_model": "tokenizer/eng.model", "english_vocab": "tokenizer/eng.vocab",
        },
        "files": exported,
        "patch_applied": patch_applied,
        "patch_equivalence_max_abs_diff": patch_diff,
        "verification": verification,
    }
    meta_path = os.path.join(out_dir, "export_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    LOGGER.info("导出完成，共 %d 个 ONNX 文件，输出目录: %s", len(exported), out_dir)
    for item in exported:
        LOGGER.info("  - %s (%.1f MB, %s)", os.path.basename(item["file"]),
                    os.path.getsize(item["file"]) / 1024 / 1024, item["backend"])
    LOGGER.info("  - %s", os.path.basename(meta_path))
    return 0


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
