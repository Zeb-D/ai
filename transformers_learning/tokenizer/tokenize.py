# 该库 用于无监督训练子词 （BPM/Unigram）模型以及后续编解码器
import os

import sentencepiece as spm


def train(input_file, vocab_size, model_name, model_type, character_coverage):
    print(os.getcwd())

    args = [
        f"--input={input_file}",
        f"--model_prefix={model_name}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--character_coverage={character_coverage}",
        "--pad_id=0",
        "--unk_id=1",
        "--bos_id=2",
        "--eos_id=3"
    ]
    cmd = " ".join(args)

    # 开始训练，会在当前目录下生成 <model_name>.model / <model_name>.vocab
    spm.SentencePieceTrainer.Train(cmd)


def run():
    # 英文分词器配置
    en_input = '../data/corpus.en'
    en_vocab_size = 32000  # 词表在翻译任务常见 16k/32k
    en_model_name = 'eng'  # 如后缀名称 eng.model / eng.vocab
    en_model_type = 'bpe'  # 使用 BPM 也可以使用 unigram
    en_character_coverage = 1.0  # 英文字符集小

    train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)

    # 中文分词器配置
    ch_input = '../data/corpus.ch'  # 一行一句话 不需要分词
    ch_vocab_size = 32000
    ch_model_name = 'chn'
    ch_model_type = 'bpe'
    ch_character_coverage = 0.9995  # 中文推荐 0.9995，极少数冷僻字会映射为 <unk>

    train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)


def test1():
    # 加载并调用己训练好的模型进行编码/解码的示例
    sp = spm.SentencePieceProcessor()
    text = "美国总统特朗普今日抵达夏威夷。"

    # 加载中文模型(确保 chn.model 位于当前工作目录)
    # .model是“分词器的大脑’ 个二进制模型，包含如何把字符串切成子词、如何把子词变回文本，以及所有规则与ID 映射
    sp.load("./chn.model")

    # 编码为子词片段(字符串)，如[’美国"，'总统'，...]
    print(sp.EncodeAsPieces(text))

    # 编码为 id(整数序列)
    print(sp.EncodeAsIds(text))

    # 示例:给定一串 id，解码回文本
    a = [12907, 277, 7419, 7318]


if __name__ == "__main__":
    # run()
    test1()
