# coding: utf-8
"""构建 / 加载辅助函数（仅本工程内部使用，**不会**随模型仓库发布）。

这里把 ``config.py`` 里的超参与标准 ``TransformerConfig`` 对接，并提供"从历史 ``.pth``
加载到新 HF 模型"的能力，供 ``train_hf.py`` / ``push_to_hub.py`` 共用。
发布到 Hub 的仓库只包含 ``configuration_transformer.py`` / ``modeling_transformer.py`` /
``tokenization_transformer.py``，与这些工程内辅助函数无关。
"""

from __future__ import annotations

import os
from typing import Optional

import torch

import config
from .configuration_transformer import TransformerConfig
from .modeling_transformer import TransformerForConditionalGeneration


def build_transformer_config(tokenizer=None, **overrides) -> TransformerConfig:
    """根据 ``config.py``（可选地用 tokenizer 的词表大小校正）构造 ``TransformerConfig``。"""
    src_vocab = int(getattr(config, "src_vocab_size", 32000))
    tgt_vocab = int(getattr(config, "tgt_vocab_size", 32000))
    if tokenizer is not None:
        src_vocab = int(tokenizer.vocab_size)
        tgt_vocab = int(len(tokenizer.get_tgt_vocab()))

    params = {
        "d_model": int(config.d_model),
        "encoder_layers": int(config.n_layers),
        "decoder_layers": int(config.n_layers),
        "encoder_attention_heads": int(config.n_heads),
        "decoder_attention_heads": int(config.n_heads),
        "encoder_ffn_dim": int(config.d_ff),
        "decoder_ffn_dim": int(config.d_ff),
        "dropout": float(config.dropout),
        "attention_dropout": float(config.dropout),
        "activation_dropout": float(config.dropout),
        "activation_function": "relu",
        "max_position_embeddings": 5000,
        "scale_embedding": True,
        "pre_norm": False,
        "src_vocab_size": src_vocab,
        "tgt_vocab_size": tgt_vocab,
        "vocab_size": tgt_vocab,
        "pad_token_id": int(config.padding_idx),
        "unk_token_id": 1,
        "bos_token_id": int(config.bos_idx),
        "eos_token_id": int(config.eos_idx),
        "decoder_start_token_id": int(config.bos_idx),
    }
    params.update(overrides)
    return TransformerConfig(**params)


def load_legacy_state_dict(path: str, map_location: str = "cpu") -> dict:
    """读取历史 ``.pth``；兼容 ``{'state_dict': ...}`` 包装与 ``DataParallel`` 的 ``module.`` 前缀。"""
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:  # noqa: BLE001 - 某些旧权重含非张量对象，退回非严格模式
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

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


def load_model_from_checkpoint(ckpt_path: str, tokenizer=None,
                               config_overrides: Optional[dict] = None
                               ) -> tuple[TransformerForConditionalGeneration, dict]:
    """构造 HF 模型并加载历史 ``.pth`` 权重（键名与之完全一致，无需任何映射）。

    返回 ``(model, report)``，report 里含缺失/多余键名，便于排查。
    """
    cfg = build_transformer_config(tokenizer=tokenizer, **(config_overrides or {}))
    model = TransformerForConditionalGeneration(cfg)

    state_dict = load_legacy_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # pe 是持久 buffer，理应被加载；其余缺失键说明结构不匹配
    missing = [k for k in missing if not k.endswith(".pe")]
    report = {
        "checkpoint": os.path.abspath(ckpt_path),
        "num_tensors": len(state_dict),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    if missing or unexpected:
        raise RuntimeError(
            f"权重与模型结构不匹配：缺失 {len(missing)} 个键，多余 {len(unexpected)} 个键。"
            f"缺失示例={missing[:5]}，多余示例={unexpected[:5]}"
        )
    model.eval()
    return model, report


__all__ = ["build_transformer_config", "load_legacy_state_dict", "load_model_from_checkpoint"]
