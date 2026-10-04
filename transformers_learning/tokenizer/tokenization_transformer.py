# coding: utf-8
"""HuggingFace 分词器：``TransformerTokenizer``。

本工程是英译中任务，源语言（英文）与目标语言（中文）各自使用**独立的 SentencePiece 模型**，
这正是 HuggingFace ``MarianTokenizer`` 所支持的标准场景（``separate_vocabs=True``）。
因此这里直接继承 ``MarianTokenizer``（HF 官方针对 en→xx 翻译任务的内置分词器），
只做两点必要扩展：

1. **源端补 BOS**：训练时源句为 ``[BOS] + pieces + [EOS]``，而 ``MarianTokenizer`` 默认只加 EOS；
   这里重写 ``build_inputs_with_special_tokens``，在 **input（源端）模式**下补 BOS。
2. **目标端只加 EOS**：训练时 labels 为 ``pieces + [EOS]``（BOS 由模型 ``_shift_right`` 用
   ``decoder_start_token_id`` 补在第一格），因此 target（目标端）模式只加 EOS。

判别源/目标端的方式沿用 ``MarianTokenizer`` 自身的模式切换（``current_spm is spm_target``），
与 ``tokenizer(text=..., text_target=...)`` 的官方用法完全一致：

    tok = AutoTokenizer.from_pretrained("your-name/transformer-en-zh", trust_remote_code=True)
    batch = tok(["The cat is sleeping."], text_target=["猫在睡觉。"], return_tensors="pt")
    # batch["input_ids"] -> <s> ... </s>
    # batch["labels"]    ->      ... </s>

仓库里随分词器一起分发的文件：``vocab.json`` / ``target_vocab.json`` /
``source.spm`` / ``target.spm``（由 ``save_pretrained`` 自动写出）。
"""

from __future__ import annotations

import json
import os

from transformers import MarianTokenizer

__all__ = ["TransformerTokenizer", "build_vocab_files"]


def build_vocab_files(source_spm: str, target_spm: str, output_dir: str) -> dict:
    """由两个 SentencePiece 模型生成 HF 需要的 ``vocab.json`` / ``target_vocab.json``。

    HF 的 ``MarianTokenizer`` 用 ``{piece: id}`` 的 JSON 词表把子词映射到 id；
    这里直接按 SentencePiece 的 ``id_to_piece`` 顺序导出，保证 token<->id 与 spm 完全一致。
    """
    import sentencepiece as spm

    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for model_path, filename in ((source_spm, "vocab.json"), (target_spm, "target_vocab.json")):
        proc = spm.SentencePieceProcessor()
        proc.Load(model_path)
        vocab = {proc.id_to_piece(i): i for i in range(proc.GetPieceSize())}
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(vocab, fp, ensure_ascii=False)
        paths[filename] = out_path
    return paths


class TransformerTokenizer(MarianTokenizer):
    """源/目标独立词表的 Marian 分词器：源端补 BOS，目标端仅补 EOS。"""

    vocab_files_names = {**MarianTokenizer.vocab_files_names}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        source_spm: str,
        target_spm: str,
        vocab: str,
        target_vocab_file: str | None = None,
        source_lang: str = "en",
        target_lang: str = "zh",
        unk_token: str = "<unk>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        model_max_length: int = 512,
        separate_vocabs: bool = True,
        sp_model_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        if separate_vocabs and target_vocab_file is None:
            raise ValueError("separate_vocabs=True 时必须提供 target_vocab_file")
        super().__init__(
            source_spm=source_spm,
            target_spm=target_spm,
            vocab=vocab,
            target_vocab_file=target_vocab_file,
            source_lang=source_lang,
            target_lang=target_lang,
            unk_token=unk_token,
            eos_token=eos_token,
            pad_token=pad_token,
            model_max_length=model_max_length,
            sp_model_kwargs=sp_model_kwargs,
            separate_vocabs=separate_vocabs,
            bos_token=bos_token,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 本地构建便捷入口（首次从 tokenizer/eng.model + chn.model 构造时使用）
    # ------------------------------------------------------------------
    @classmethod
    def from_sentencepiece(
        cls,
        source_spm: str,
        target_spm: str,
        vocab_dir: str | None = None,
        **kwargs,
    ) -> "TransformerTokenizer":
        """由 ``source_spm`` / ``target_spm`` 生成词表文件并构造分词器。"""
        vocab_dir = vocab_dir or os.path.dirname(os.path.abspath(source_spm)) or "."
        paths = build_vocab_files(source_spm, target_spm, vocab_dir)
        return cls(
            source_spm=str(source_spm),
            target_spm=str(target_spm),
            vocab=paths["vocab.json"],
            target_vocab_file=paths["target_vocab.json"],
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 源端 / 目标端的 special token 规则
    # ------------------------------------------------------------------
    def _in_target_mode(self) -> bool:
        """当前是否处于目标端（解码端）编码上下文。"""
        return bool(getattr(self, "separate_vocabs", False)) and self.current_spm is self.spm_target

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """源端：``[BOS] + tokens + [EOS]``；目标端：``tokens + [EOS]``。"""
        prefix = [] if self._in_target_mode() else [self.bos_token_id]
        if token_ids_1 is None:
            return prefix + list(token_ids_0) + [self.eos_token_id]
        return prefix + list(token_ids_0) + list(token_ids_1) + [self.eos_token_id]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 1 if self._in_target_mode() else 2

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None,
                                already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, True)
        length = len(token_ids_0) + (len(token_ids_1) if token_ids_1 else 0)
        if self._in_target_mode():
            return [0] * length + [1]
        return [1] + [0] * length + [1]

    # ------------------------------------------------------------------
    # 词表访问（覆盖上游 MarianTokenizer 对 added_tokens_decoder 的错误展开，
    # 该实现用 ``dict(vocab, **{int: token})`` 会因 int 键报 "keywords must be strings"）
    # ------------------------------------------------------------------
    def get_vocab(self) -> dict:
        return self.get_src_vocab()

    def get_src_vocab(self) -> dict:
        vocab = dict(self.encoder)
        for idx, token in self.added_tokens_decoder.items():
            vocab[str(token)] = idx
        return vocab

    def get_tgt_vocab(self) -> dict:
        vocab = dict(self.target_encoder)
        for idx, token in self.added_tokens_decoder.items():
            vocab[str(token)] = idx
        return vocab
