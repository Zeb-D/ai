import os

import sentencepiece as spm

# 分词模型与 HF 词表文件的默认位置（相对于 transformers_learning 目录）
DEFAULT_SOURCE_SPM = "./tokenizer/eng.model"
DEFAULT_TARGET_SPM = "./tokenizer/chn.model"


def chinese_tokenizer_load():
    """
    加载中文分词器模型
    该函数用于加载预训练的中文SentencePiece分词器模型，用于文本预处理和分词。
    返回:
        sp_chn: 加载好的SentencePieceProcessor对象，可用于中文文本的分词处理
    使用方法:
        tokenizer = chinese_tokenizer_load()
        tokens = tokenizer.tokenize("中文文本")
    """
    # 创建SentencePieceProcessor对象
    sp_chn = spm.SentencePieceProcessor()
    # 加载预训练的中文分词模型，模型路径为"./tokenizer/chn.model"
    sp_chn.Load('{}.model'.format("./tokenizer/chn"))
    # 返回加载好的分词器对象
    return sp_chn


def english_tokenizer_load():
    """
    加载英文分词器模型
    该函数用于加载英文的SentencePiece分词器模型，该模型用于将英文文本转换为token序列。
    返回:
        SentencePieceProcessor: 加载了英文模型的SentencePieceProcessor对象，可用于英文文本的分词处理
    示例:
        tokenizer = english_tokenizer_load()
        tokens = tokenizer.encode("Hello world")
    """
    # 创建SentencePieceProcessor对象
    sp_eng = spm.SentencePieceProcessor()
    # 加载预训练的英文分词模型，模型路径为"./tokenizer/eng.model"
    sp_eng.Load('{}.model'.format("./tokenizer/eng"))
    # 返回加载好的分词器对象
    return sp_eng


def hf_tokenizer_load(source_spm: str = DEFAULT_SOURCE_SPM,
                      target_spm: str = DEFAULT_TARGET_SPM,
                      max_length: int = None,
                      vocab_dir: str = None,
                      **kwargs):
    """加载标准 HuggingFace 分词器（``TransformerTokenizer``）。

    与上面的 ``*_tokenizer_load`` 不同，这里返回的是真正的 ``PreTrainedTokenizer``：
    它同时用英文 spm 编码源句、用中文 spm 解码目标句，支持 ``text_target``、
    ``save_pretrained`` / ``push_to_hub``，是训练与发布链路统一使用的分词器。

    首次调用会在 ``vocab_dir``（默认分词模型所在目录）生成 ``vocab.json`` /
    ``target_vocab.json`` 两个 HF 词表文件。
    """
    from tokenizer.tokenization_transformer import TransformerTokenizer

    if max_length is not None:
        kwargs.setdefault("model_max_length", int(max_length))
    vocab_dir = vocab_dir or os.path.dirname(os.path.abspath(source_spm))
    return TransformerTokenizer.from_sentencepiece(source_spm, target_spm,
                                                   vocab_dir=vocab_dir, **kwargs)