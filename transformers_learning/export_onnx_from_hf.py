#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 HuggingFace 上游模型导出为 ONNX（默认 ``chou-lucas/transformer-en-zh``）。

加载方式与 [`hf_inference.py`](hf_inference.py:1) 完全一致：
``AutoTokenizer`` + ``AutoModelForSeq2SeqLM``，默认**不需要** ``trust_remote_code``，
所以导出的就是"别人拿到 HF 仓库后能直接用"的那个模型。

由于 transformers>=5 已移除内置 ONNX 导出（迁到 optimum），而当前环境没有 optimum，
本脚本用 ``torch.onnx.export`` 手工导出标准 seq2seq 的两段子图：

    encoder_model.onnx   encoder(input_ids, attention_mask) -> last_hidden_state
    decoder_model.onnx   decoder(encoder_hidden_states, encoder_attention_mask,
                                 decoder_input_ids) -> logits

生成（beam search）由 [`hf_onnx_inference.py`](hf_onnx_inference.py:1) 在
onnxruntime 上完成，无需 past_key_values，短句翻译足够快。

用法::

    python export_onnx_from_hf.py \
        --model chou-lucas/transformer-en-zh \
        --out   transformers_learning/onnx_model

国内网络直连 huggingface.co 会超时，可改用镜像::

    HF_ENDPOINT=https://hf-mirror.com python export_onnx_from_hf.py \
        --model chou-lucas/transformer-en-zh --out transformers_learning/onnx_model

依赖：``transformers>=5 torch onnx onnxruntime sentencepiece sacremoses``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

# 脚本所在目录：默认输出目录相对它解析，避免受运行时 CWD 影响
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="export_onnx_from_hf.py",
        description="把 HF 上游模型（默认 chou-lucas/transformer-en-zh）导出为 ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="chou-lucas/transformer-en-zh",
                        help="HF 仓库 id 或本地目录（与 hf_inference.py 一致）")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "onnx_model"),
                        help="ONNX 输出目录")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset 版本")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="仅当模型仓库是自定义架构（带 .py）时才需要")
    parser.add_argument("--batch-size", type=int, default=2, help="导出用 dummy batch 大小")
    parser.add_argument("--src-len", type=int, default=8, help="导出用 dummy 源长度")
    parser.add_argument("--dec-len", type=int, default=5, help="导出用 dummy 目标长度")
    return parser.parse_args(argv)


def load_model(model_id: str, trust_remote_code: bool):
    """按 HF 标准方式加载（与 hf_inference.py 相同）。"""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                                  trust_remote_code=trust_remote_code)
    model.eval()
    return tokenizer, model


class EncoderWrapper(nn.Module):
    """encoder(input_ids, attention_mask) -> last_hidden_state"""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.get_encoder()

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]


class DecoderWrapper(nn.Module):
    """decoder(encoder_hidden_states, encoder_attention_mask, decoder_input_ids) -> logits"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, encoder_hidden_states, encoder_attention_mask, decoder_input_ids):
        from transformers.modeling_outputs import BaseModelOutput

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        out = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,   # seq2seq 中即 encoder attention mask
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


def _export(module, args, output_path, input_names, output_names,
            dynamic_axes, opset):
    kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    # torch>=2.6 默认 dynamo=True，这里强制走稳定的 TorchScript 导出器。
    try:
        torch.onnx.export(module, args, output_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(module, args, output_path, **kwargs)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 加载 HF 模型：{args.model}")
    tokenizer, model = load_model(args.model, args.trust_remote_code)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      tokenizer={type(tokenizer).__name__}  model={type(model).__name__}  "
          f"model_type={model.config.model_type}  params={n_params:.1f}M  vocab={model.config.vocab_size}")

    # ---- dummy 输入 ----
    B, S, T = args.batch_size, args.src_len, args.dec_len
    vocab = int(model.config.vocab_size)
    pad_id = int(model.config.pad_token_id or 0)
    start_id = int(
        getattr(model.generation_config, "decoder_start_token_id", None)
        or model.config.decoder_start_token_id
        or pad_id
    )
    enc_ids = torch.randint(low=3, high=max(4, vocab - 1), size=(B, S), dtype=torch.long)
    enc_mask = torch.ones((B, S), dtype=torch.long)
    enc_mask[:, -1] = 0  # 制造一个 padding，让 attention_mask 参与计算
    dec_ids = torch.full((B, T), start_id, dtype=torch.long)

    encoder = EncoderWrapper(model).eval()
    decoder = DecoderWrapper(model).eval()

    with torch.no_grad():
        hidden = encoder(enc_ids, enc_mask)
    print(f"      编码器输出 last_hidden_state: {tuple(hidden.shape)}")

    # ---- 导出 encoder ----
    enc_path = out_dir / "encoder_model.onnx"
    print(f"[2/3] 导出 encoder -> {enc_path}")
    _export(
        encoder,
        (enc_ids, enc_mask),
        str(enc_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "src_len"},
            "attention_mask": {0: "batch", 1: "src_len"},
            "last_hidden_state": {0: "batch", 1: "src_len"},
        },
        opset=args.opset,
    )

    # ---- 导出 decoder ----
    dec_path = out_dir / "decoder_model.onnx"
    print(f"      导出 decoder -> {dec_path}")
    with torch.no_grad():
        _export(
            decoder,
            (hidden, enc_mask, dec_ids),
            str(dec_path),
            input_names=["encoder_hidden_states", "encoder_attention_mask", "decoder_input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "encoder_hidden_states": {0: "batch", 1: "src_len"},
                "encoder_attention_mask": {0: "batch", 1: "src_len"},
                "decoder_input_ids": {0: "batch", 1: "tgt_len"},
                "logits": {0: "batch", 1: "tgt_len"},
            },
            opset=args.opset,
        )

    # ---- 保存 tokenizer / config / 元信息 ----
    print(f"[3/3] 保存 tokenizer、config 与元信息 -> {out_dir}")
    tokenizer.save_pretrained(out_dir)
    model.config.save_pretrained(out_dir)
    try:
        model.generation_config.save_pretrained(out_dir)
    except Exception as exc:  # pragma: no cover
        print(f"      (generation_config 未保存：{exc})")

    meta = {
        "source_model": args.model,
        "model_type": model.config.model_type,
        "vocab_size": vocab,
        "decoder_start_token_id": start_id,
        "eos_token_id": int(getattr(model.generation_config, "eos_token_id", None)
                            or model.config.eos_token_id),
        "pad_token_id": pad_id,
        "bos_token_id": int(getattr(model.generation_config, "bos_token_id", None)
                            or model.config.bos_token_id or start_id),
        "hidden_size": int(getattr(model.config, "d_model", 0) or getattr(model.config, "hidden_size", 0)),
        "opset": args.opset,
        "files": ["encoder_model.onnx", "decoder_model.onnx"],
    }
    (out_dir / "onnx_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for f in sorted(out_dir.iterdir()):
        size = f.stat().st_size / 1e6 if f.is_file() else 0
        print(f"      {f.name:<28} {size:7.2f} MB" if f.is_file() else f"      {f.name}/")
    print("完成 ✅ 用 hf_onnx_inference.py 验证效果。")
    return 0


if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    sys.exit(main())
