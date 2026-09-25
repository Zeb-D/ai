# coding: utf-8
"""HuggingFace 自定义分词器（随模型仓库发布，配合 trust_remote_code=True 使用）。

为什么需要它：
    1. 本工程的源语言（英文）和目标语言（中文）用的是**两套独立**的 SentencePiece 模型，
       而 transformers 的 MarianTokenizer 只有在 separate_vocabs=True 时才会用
       target.spm 去 decode。本类固定 separate_vocabs=True（配置里也会写上），
       保证 "encode 用 source.spm / decode 用 target.spm" 与训练时一致。
    2. MarianTokenizer 默认只在句尾加 EOS，而本模型训练时用的是 [BOS] + pieces + [EOS]，
       所以这里覆盖 build_inputs_with_special_tokens，在句首补 BOS。

仓库里配套的文件：source.spm / target.spm / vocab.json / target_vocab.json / tokenizer_config.json

用法：

    tokenizer = AutoTokenizer.from_pretrained("your-name/your-repo", trust_remote_code=True)
    tokenizer("The cat is sleeping on the sofa.")["input_ids"]   # [2, 99, 2514, ..., 3]
"""

from __future__ import annotations

from transformers.models.marian.tokenization_marian import MarianTokenizer


class TransformerCustomTokenizer(MarianTokenizer):
    """MarianTokenizer 的薄封装：源/目标词表分离 + 句首补 BOS。"""

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.bos_token_id] + list(token_ids_0) + [self.eos_token_id]
        return ([self.bos_token_id] + list(token_ids_0) + list(token_ids_1)
                + [self.eos_token_id])


__all__ = ["TransformerCustomTokenizer"]
