# coding: utf-8
"""HuggingFace 配置类：``TransformerConfig``。

本文件是"标准 HF 方式"的单一事实来源（single source of truth），既用于本地训练/推理，
也会由 ``save_pretrained`` 自动拷贝进模型仓库，配合 ``trust_remote_code=True`` 供他人加载：

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("your-name/transformer-en-zh", trust_remote_code=True)

字段命名对齐 HuggingFace 的 seq2seq 惯例（``encoder_layers`` / ``decoder_attention_heads`` …），
因此与 ``AutoConfig`` / ``AutoModelForSeq2SeqLM`` 的自动推断逻辑完全兼容。

架构语义（与 ``model/modeling_transformer.py`` 严格一一对应）：

* Post-LN：``x + dropout(sublayer(norm(x)))``（``pre_norm=False``）
* Embedding 乘 ``sqrt(d_model)``（``scale_embedding=True``）
* 固定 sin-cos 位置编码（encoder / decoder 各一份 buffer）
* 源语言与目标语言各自独立词表（``src_vocab_size`` / ``tgt_vocab_size``）
"""

from __future__ import annotations

from transformers import PretrainedConfig


class TransformerConfig(PretrainedConfig):
    """自定义 Transformer（encoder-decoder）的结构配置。"""

    model_type = "transformer_custom"
    # 训练时不需要推理期的 past_key_values，避免 Agent 相关逻辑误读
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        src_vocab_size: int = 32000,
        tgt_vocab_size: int = 32000,
        d_model: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        encoder_attention_heads: int = 8,
        decoder_attention_heads: int = 8,
        encoder_ffn_dim: int = 2048,
        decoder_ffn_dim: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_function: str = "relu",
        max_position_embeddings: int = 5000,
        scale_embedding: bool = True,
        pre_norm: bool = False,
        unk_token_id: int = 1,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        decoder_start_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = int(vocab_size)
        self.src_vocab_size = int(src_vocab_size)
        self.tgt_vocab_size = int(tgt_vocab_size)
        self.d_model = int(d_model)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.encoder_attention_heads = int(encoder_attention_heads)
        self.decoder_attention_heads = int(decoder_attention_heads)
        self.encoder_ffn_dim = int(encoder_ffn_dim)
        self.decoder_ffn_dim = int(decoder_ffn_dim)
        self.dropout = float(dropout)
        self.attention_dropout = float(attention_dropout)
        self.activation_dropout = float(activation_dropout)
        self.activation_function = activation_function
        self.max_position_embeddings = int(max_position_embeddings)
        self.scale_embedding = bool(scale_embedding)
        self.pre_norm = bool(pre_norm)
        self.unk_token_id = int(unk_token_id)
        # 一些通用工具（显存估算 / 静态 shape 推断）会读取 num_hidden_layers
        self.num_hidden_layers = int(encoder_layers)

        # 这三个字段由 PretrainedConfig 统一接管；decoder 起始符默认复用 BOS
        kwargs.pop("is_encoder_decoder", None)
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            pad_token_id=int(pad_token_id),
            bos_token_id=int(bos_token_id),
            eos_token_id=int(eos_token_id),
            decoder_start_token_id=(
                int(decoder_start_token_id)
                if decoder_start_token_id is not None
                else int(bos_token_id)
            ),
            is_encoder_decoder=True,
            # src_embed / tgt_embed / generator 三者权重互不共享
            tie_word_embeddings=False,
            **kwargs,
        )


__all__ = ["TransformerConfig"]
