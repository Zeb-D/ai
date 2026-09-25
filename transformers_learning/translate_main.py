import torch
import config
import logging
import numpy as np

from tools.beam_decoder import beam_search
from tools.tokenizer_utils import english_tokenizer_load
from model.tf_model import make_model
from tools.tokenizer_utils import chinese_tokenizer_load

logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s-%(funcName)s:%(lineno)d', level=logging.INFO)


def translate(src, model, chn_tokenizer):
    """用训练好的模型进行预测单句，打印模型翻译结果"""

    # 注意：模型和分词器已在外部预先加载，这里不再重复加载，避免每次翻译都从磁盘读取。
    # 原代码中在此处加载中文分词器和模型权重，现已移至 translate_example 中统一初始化。

    with torch.no_grad():  # 禁用梯度计算，以节省内存
        # 模型已设置为评估模式，这里再次确保（也可以省略）
        model.eval()

        # 创建源句子的掩码（mask），以确保填充的部分不会参与计算
        src_mask = (src != 0).unsqueeze(-2)

        # 使用束搜索（beam search）进行解码
        decode_result, _ = beam_search(
            model,
            src,
            src_mask,
            config.max_len,  # 最大翻译长度
            config.padding_idx,  # 填充符号的索引
            config.bos_idx,  # 句子开始符号的索引
            config.eos_idx,  # 句子结束符号的索引
            config.beam_size,  # 束搜索的大小
            config.device  # 设备（CPU或GPU）
        )

        # 从解码结果中提取最优结果
        decode_result = [h[0] for h in decode_result]

        # 使用中文分词器将解码结果的id转化为实际的中文词语
        translation = [chn_tokenizer.decode_ids(_s) for _s in decode_result]

        # # 打印并返回翻译结果的第一句
        # print(translation[0])
        return translation[0]


def one_sentence_translate(sent, model, en_tokenizer, chn_tokenizer, BOS, EOS):
    """翻译单句英文"""

    # 注意：模型和分词器已作为参数传入，不再在函数内部重复创建和加载。
    # 原代码中在此处初始化模型、加载分词器等操作已移至 translate_example 中，只执行一次。

    # 将输入的句子转化为token IDs，添加BOS和EOS标记
    src_tokens = [[BOS] + en_tokenizer.EncodeAsIds(sent) + [EOS]]

    # 将句子转换为长整型Tensor，并发送到指定的设备（如GPU或CPU）
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)

    # 调用translate函数进行翻译
    return translate(batch_input, model, chn_tokenizer)


def translate_example():
    """单句翻译示例"""
    # 示例句子（原代码中的孤立字符串，保留）
    "The government has implemented various policies to improve the living standards of its citizens."
    "政府实施了诸多政策，改善公民的生活水平。"

    # ========== 性能优化：预先加载模型、权重和分词器（只执行一次） ==========
    # 1. 加载分词器
    en_tokenizer = english_tokenizer_load()
    chn_tokenizer = chinese_tokenizer_load()
    BOS = en_tokenizer.bos_id()  # 获取开始符号（BOS）的ID，通常是2
    EOS = en_tokenizer.eos_id()  # 获取结束符号（EOS）的ID，通常是3

    # 2. 构建模型并加载权重
    model = make_model(
        config.src_vocab_size,  # 源语言词汇表大小
        config.tgt_vocab_size,  # 目标语言词汇表大小
        config.n_layers,  # 模型的层数
        config.d_model,  # 模型的维度（通常是隐藏层的大小）
        config.d_ff,  # 前馈网络的维度
        config.n_heads,  # 注意力头的数量
        config.dropout  # dropout比率
    )
    # 加载训练好的模型权重，并移动到指定设备
    model.load_state_dict(torch.load(config.translate_model_path, map_location=config.device))
    model.to(config.device)
    model.eval()  # 设置为评估模式

    # ========== 循环翻译 ==========
    while True:  # 使用循环，让用户可以反复输入句子
        # 提示用户输入英文句子
        sent = input("请输入英文句子进行翻译（输入 q! 退出）：")

        # 判断是否退出
        if sent.strip() == "q!":
            print("已退出翻译程序。")
            break

        # 跳过空输入
        if not sent.strip():
            print("输入为空，请重新输入。")
            continue

        # 调用翻译函数进行翻译（传入预加载的模型和分词器）
        translation = one_sentence_translate(sent, model, en_tokenizer, chn_tokenizer, BOS, EOS)
        print("翻译结果：", translation)


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import warnings

    warnings.filterwarnings('ignore')
    translate_example()
