# coding: utf-8
"""HuggingFace 自定义模型定义（随模型仓库一起发布，配合 trust_remote_code=True 使用）。

加载方式（上传到 Hub 之后，别人只需要这两行）：

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("your-name/your-repo")
    model = AutoModelForSeq2SeqLM.from_pretrained("your-name/your-repo", trust_remote_code=True)
    out = model.generate(**tokenizer(["The cat is sleeping on the sofa."], return_tensors="pt"),
                         max_length=60, num_beams=3)
    print(tokenizer.batch_decode(out, skip_special_tokens=True))

要点：
    1. 参数名与本工程的 PyTorch 权重一一对应（src_embed / tgt_embed / encoder / decoder /
       generator），所以 model.safetensors 不需要做任何键名转换，from_pretrained 直接可加载。
    2. 结构与训练时完全一致：Post-LN、sin-cos 固定位置编码、embedding 乘 sqrt(d_model)、
       attention 用 -1e9 做 mask、生成器输出 logits（不做 log_softmax）。
    3. 不实现 KV Cache（每步重算整段 self-attention），因此 forward 永远返回
       past_key_values=None，与仓库里的 encoder_model.onnx / decoder_model.onnx 行为一致。
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

try:  # 从 Hub 以 trust_remote_code=True 加载时，HF 会把同目录的 .py 一起拉下来，可用相对导入
    from .configuration_transformer_custom import TransformerCustomConfig
except ImportError:  # 直接在本地 python 里 import 时退化为绝对导入
    from configuration_transformer_custom import TransformerCustomConfig


# ===========================================================================
# mask 工具：与训练侧 tools/data_loader.py 的规则完全一致
# ===========================================================================
def make_src_mask(attention_mask: Optional[Tensor]) -> Optional[Tensor]:
    """HF 风格 attention_mask (B, S) -> (B, 1, S)，供 attention 广播到 (B, h, S, S)。"""
    if attention_mask is None:
        return None
    return (attention_mask != 0).unsqueeze(-2)


def make_tgt_mask(decoder_input_ids: Tensor, padding_idx: int) -> Tensor:
    """(B, T) -> (B, T, T)：padding mask & 下三角（防止看到未来）。"""
    length = decoder_input_ids.size(1)
    positions = torch.arange(length, device=decoder_input_ids.device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    return (decoder_input_ids != padding_idx).unsqueeze(-2) & causal


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """克隆 N 个互不共享参数的模块。"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# ===========================================================================
# 基础组件（模块名/参数名与训练代码保持一致）
# ===========================================================================
class Embeddings(nn.Module):
    """词索引 -> 向量，并乘 sqrt(d_model)。参数：lut.weight"""

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """固定 sin-cos 位置编码；buffer 名称为 pe（会一起保存在 safetensors 里）。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))          # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None,
              dropout: Optional[nn.Dropout] = None) -> Tuple[Tensor, Tensor]:
    """缩放点积注意力；mask 里为 0/False 的位置填 -1e9。"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """多头注意力；参数：linears.{0..3}.weight/bias"""

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for linear, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """自定义 LayerNorm（无偏标准差）；参数：a_2 / b_2"""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    """FFN；参数：w_1.weight/bias, w_2.weight/bias"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """Post-LN + 残差；参数：norm.a_2 / norm.b_2"""

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """参数：self_attn.*, feed_forward.*, sublayer.{0,1}.norm.*"""

    def __init__(self, size: int, self_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class TransformerEncoder(nn.Module):
    """编码器堆叠（输入为 embedding 后的张量）；参数：layers.*, norm.*"""

    # generate() 里 `_prepare_model_inputs` 会读 self.encoder.main_input_name
    main_input_name = "input_ids"
    _is_encoder_decoder_encoder = True

    def __init__(self, layer: EncoderLayer, n: int):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """参数：self_attn.*, src_attn.*, feed_forward.*, sublayer.{0,1,2}.norm.*"""

    def __init__(self, size: int, self_attn: MultiHeadedAttention, src_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward, dropout: float):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x: Tensor, memory: Tensor, src_mask: Optional[Tensor],
                tgt_mask: Optional[Tensor]) -> Tensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class TransformerDecoder(nn.Module):
    """解码器堆叠（输入为 embedding 后的张量）；参数：layers.*, norm.*"""

    main_input_name = "input_ids"

    def __init__(self, layer: DecoderLayer, n: int):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, memory: Tensor, src_mask: Optional[Tensor],
                tgt_mask: Optional[Tensor]) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """输出投影；参数：proj.weight / proj.bias（输出未归一化的 logits，符合 HF 约定）"""

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(self.proj(x), dim=-1)


class _EncoderWithEmbeddings(nn.Module):
    """genreate() 会通过 get_encoder() 拿到编码器并要求 input_ids/attention_mask 入参。

    本工程的 embedding 放在模型顶层（src_embed），所以这里包一层，把 embedding 补上并返回
    HF 的 BaseModelOutput。
    """

    def __init__(self, owner: "TransformerForConditionalGeneration"):
        super().__init__()
        object.__setattr__(self, "_owner", owner)      # 不注册为子模块，避免污染参数名
        self.config = owner.config

    @property
    def owner(self) -> "TransformerForConditionalGeneration":
        return self._owner

    def forward(self, input_ids: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
                inputs_embeds: Optional[Tensor] = None, **kwargs) -> BaseModelOutput:
        owner = self.owner
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("必须提供 input_ids 或 inputs_embeds")
            inputs_embeds = owner.src_embed(input_ids)
        hidden = owner.encoder(inputs_embeds, make_src_mask(attention_mask))
        return BaseModelOutput(last_hidden_state=hidden)


# ===========================================================================
# 顶层模型
# ===========================================================================
class TransformerForConditionalGeneration(PreTrainedModel):
    """自定义 Transformer 的 seq2seq 封装，接口对齐 HuggingFace（支持 generate()）。"""

    config_class = TransformerCustomConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def __init__(self, config: TransformerCustomConfig):
        super().__init__(config)
        c = copy.deepcopy
        attn = MultiHeadedAttention(config.n_heads, config.d_model, config.dropout)
        ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        position = PositionalEncoding(config.d_model, config.dropout,
                                      config.max_position_embeddings)

        self.src_embed = nn.Sequential(Embeddings(config.d_model, config.src_vocab_size),
                                       c(position))
        self.tgt_embed = nn.Sequential(Embeddings(config.d_model, config.tgt_vocab_size),
                                       c(position))
        self.encoder = TransformerEncoder(
            EncoderLayer(config.d_model, c(attn), c(ff), config.dropout), config.n_layers)
        self.decoder = TransformerDecoder(
            DecoderLayer(config.d_model, c(attn), c(attn), c(ff), config.dropout),
            config.n_layers)
        self.generator = Generator(config.d_model, config.tgt_vocab_size)

        self.post_init()

    # ---------------- 初始化（与训练脚本 make_model 一致：>1 维参数 xavier）----------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    # ---------------- 子模块访问（HF generate 需要）----------------
    def get_encoder(self):
        return _EncoderWithEmbeddings(self)

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.src_embed[0].lut

    def set_input_embeddings(self, value):
        self.src_embed[0].lut = value

    def get_output_embeddings(self):
        return self.generator.proj

    def set_output_embeddings(self, new_embeddings):
        self.generator.proj = new_embeddings

    def _reorder_cache(self, past_key_values, beam_idx):
        # 没有实现 KV Cache，beam search 里不需要重排
        return None if past_key_values is None else past_key_values

    def _supports_default_dynamic_cache(self) -> bool:
        """本模型没有实现 KV Cache，generate() 不需要（也不应该）准备 Cache。"""
        return False

    # ---------------- 内部：编码 / 解码 ----------------
    def encode(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """token id -> encoder 输出（memory）。"""
        return self.encoder(self.src_embed(input_ids), make_src_mask(attention_mask))

    def decode(self, memory: Tensor, src_mask: Optional[Tensor], decoder_input_ids: Tensor,
               tgt_mask: Optional[Tensor] = None) -> Tensor:
        """memory + decoder token id -> decoder hidden。"""
        if tgt_mask is None:
            tgt_mask = make_tgt_mask(decoder_input_ids, self.config.pad_token_id)
        return self.decoder(self.tgt_embed(decoder_input_ids), memory, src_mask, tgt_mask)

    def _shift_right(self, input_ids: Tensor) -> Tensor:
        """labels 右移一位，首位填 decoder_start_token_id（HF 的标准做法）。"""
        pad = self.config.pad_token_id
        start = self.config.decoder_start_token_id
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[..., 1:] = input_ids[..., :-1].clone()
        shifted[..., 0] = start
        shifted.masked_fill_(input_ids == pad, pad)
        return shifted

    # ---------------- forward ----------------
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        encoder_outputs=None,
        labels: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache=None,
        past_key_values=None,
        cache_position=None,
        return_dict=None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if return_dict is None:
            return_dict = getattr(self.config, "return_dict", None)
        if return_dict is None:
            return_dict = True

        # ---- 编码（generate() 会把 encoder_outputs 缓存起来，避免每步重算）----
        if encoder_outputs is None:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("forward 需要 input_ids / inputs_embeds / encoder_outputs 之一")
                inputs_embeds = self.src_embed(input_ids)
            memory = self.encoder(inputs_embeds, make_src_mask(attention_mask))
            encoder_outputs = BaseModelOutput(last_hidden_state=memory)
        elif isinstance(encoder_outputs, (tuple, list)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0])
        memory = encoder_outputs.last_hidden_state
        src_mask = make_src_mask(attention_mask)

        # ---- 解码 ----
        if decoder_input_ids is None and decoder_inputs_embeds is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        logits = None
        if decoder_input_ids is not None or decoder_inputs_embeds is not None:
            if decoder_inputs_embeds is None:
                decoder_inputs_embeds = self.tgt_embed(decoder_input_ids)
            tgt_mask = (make_tgt_mask(decoder_input_ids, self.config.pad_token_id)
                        if decoder_input_ids is not None else None)
            hidden = self.decoder(decoder_inputs_embeds, memory, src_mask, tgt_mask)
            logits = self.generator.proj(hidden)      # HF 约定：输出 logits

        loss = None
        if labels is not None and logits is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1),
                                   ignore_index=self.config.pad_token_id)

        if not return_dict:
            output = (logits, memory)
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,                     # 没有 KV Cache
            encoder_last_hidden_state=memory,
        )

    # ---------------- generate() 需要的钩子 ----------------
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      encoder_outputs=None, use_cache=None, **kwargs):
        return {
            "input_ids": None,                        # encoder 已经跑完，用 encoder_outputs 即可
            "attention_mask": attention_mask,
            "decoder_input_ids": input_ids,
            "encoder_outputs": encoder_outputs,
            "use_cache": False,
        }


__all__ = [
    "TransformerCustomConfig",
    "TransformerForConditionalGeneration",
]
