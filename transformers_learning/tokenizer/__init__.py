# coding: utf-8
"""标准 HuggingFace 分词器包。

``TransformerTokenizer`` 继承自 ``MarianTokenizer``，因此保存/加载、``text_target``、
``save_vocabulary`` 等全部沿用 HF 官方流程；这里额外调用 ``register_for_auto_class``，
让 ``save_pretrained`` 自动把 ``tokenization_transformer.py`` 写入模型仓库。
"""

from __future__ import annotations

from .tokenization_transformer import TransformerTokenizer, build_vocab_files

TransformerTokenizer.register_for_auto_class("AutoTokenizer")

__all__ = ["TransformerTokenizer", "build_vocab_files"]
