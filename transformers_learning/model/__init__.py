# coding: utf-8
"""标准 HuggingFace 模型包。

导入本包时会：

1. 把 ``TransformerConfig`` / ``TransformerForConditionalGeneration`` 注册进
   ``AutoConfig`` / ``AutoModelForSeq2SeqLM``，因此**本地**无需 ``trust_remote_code``
   也能用 Auto 类加载；
2. 调用 ``register_for_auto_class``，这让 ``save_pretrained`` 自动把
   ``configuration_transformer.py`` / ``modeling_transformer.py`` 写入模型仓库，
   从而他人 ``from_pretrained(..., trust_remote_code=True)`` 即可使用（HF 官方自定义模型机制）。
"""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from .configuration_transformer import TransformerConfig
from .modeling_transformer import TransformerForConditionalGeneration

# ---- 本地 Auto 映射注册 ----
AutoConfig.register(TransformerConfig.model_type, TransformerConfig, exist_ok=True)
AutoModelForSeq2SeqLM.register(TransformerConfig, TransformerForConditionalGeneration, exist_ok=True)

# ---- 声明"自定义代码"归属的 Auto 类（save_pretrained 据此拷贝源码 + 写 auto_map）----
TransformerConfig.register_for_auto_class("AutoConfig")
TransformerForConditionalGeneration.register_for_auto_class("AutoModelForSeq2SeqLM")

# ---- 让 AutoTokenizer 也能在本地自动匹配到自定义分词器 ----
try:
    from tokenizer.tokenization_transformer import TransformerTokenizer

    TransformerTokenizer.register_for_auto_class("AutoTokenizer")
    AutoTokenizer.register(
        TransformerConfig, slow_tokenizer_class=TransformerTokenizer, exist_ok=True
    )
except Exception:  # noqa: BLE001 - 分词器非必需依赖，缺失时不影响模型加载
    pass

__all__ = ["TransformerConfig", "TransformerForConditionalGeneration"]
