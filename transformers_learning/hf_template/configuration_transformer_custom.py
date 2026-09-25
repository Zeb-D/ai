# coding: utf-8
"""HuggingFace 自定义模型配置（随模型仓库一起发布，配合 trust_remote_code=True 使用）。

对应仓库里的 config.json：

    {
      "model_type": "transformer_custom",
      "architectures": ["TransformerForConditionalGeneration"],
      "auto_map": {
        "AutoConfig": "configuration_transformer_custom.TransformerCustomConfig",
        "AutoModelForSeq2SeqLM": "modeling_transformer_custom.TransformerForConditionalGeneration"
      },
      ...
    }

本文件与 modeling_transformer_custom.py 一一对应，描述的是
"标准 Transformer（Post-LN / 乘法式 embedding 缩放 / 学习式位置编码由固定 sin-cos buffer 提供）"
这一自定义架构。
"""

from __future__ import annotations

from transformers import PretrainedConfig


class TransformerCustomConfig(PretrainedConfig):
    """自定义 Transformer 的结构配置（字段命名对齐 HuggingFace 的 seq2seq 惯例）。"""

    model_type = "transformer_custom"
    # 推理时不需要输出这些字段
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        src_vocab_size: int = 32000,
        tgt_vocab_size: int = 32000,
        max_position_embeddings: int = 5000,
        activation_function: str = "relu",
        pre_norm: bool = False,
        scale_embedding: bool = True,
        unk_token_id: int = 1,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        decoder_start_token_id: int = None,
        **kwargs,
    ):
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.src_vocab_size = int(src_vocab_size)
        self.tgt_vocab_size = int(tgt_vocab_size)
        self.max_position_embeddings = int(max_position_embeddings)
        self.activation_function = activation_function
        # 便于 HF 内部（KVCache 准备、静态 shape 检查等）识别层数
        self.num_hidden_layers = int(n_layers)
        # 本工程是 Post-LN：x + dropout(sublayer(norm(x)))
        self.pre_norm = bool(pre_norm)
        # Embeddings.forward 里乘了 sqrt(d_model)
        self.scale_embedding = bool(scale_embedding)
        self.unk_token_id = int(unk_token_id)

        # 这三个由 PretrainedConfig 统一接管；decoder 起始符默认用 BOS
        kwargs.pop("is_encoder_decoder", None)
        kwargs.pop("tie_word_embeddings", None)
        kwargs.pop("vocab_size", None)
        super().__init__(
            pad_token_id=int(pad_token_id),
            bos_token_id=int(bos_token_id),
            eos_token_id=int(eos_token_id),
            decoder_start_token_id=(int(decoder_start_token_id)
                                    if decoder_start_token_id is not None
                                    else int(bos_token_id)),
            is_encoder_decoder=True,
            # src_embed / tgt_embed / generator 三者权重互不共享
            tie_word_embeddings=False,
            # 供 AutoModel 做 embedding 相关推断时使用（目标词表 == vocab_size）
            vocab_size=int(tgt_vocab_size),
            **kwargs,
        )


__all__ = ["TransformerCustomConfig"]
