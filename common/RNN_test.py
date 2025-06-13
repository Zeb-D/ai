import zipfile

import jieba
import torch
import torch.nn as nn


# 构建词典
def build_vocab():
    file_name = '../deepLearning/jaychou_lyrics.txt.zip'
    with zipfile.ZipFile('../deepLearning/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    print(len(corpus_chars))

    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


class LyricsDataset:

    def __init__(self, corpus_idx, num_chars):
        # 语料数据
        self.corpus_idx = corpus_idx
        # 语料长度
        self.num_chars = num_chars
        # 词的数量
        self.word_count = len(self.corpus_idx)
        # 句子数量
        self.number = self.word_count // self.num_chars

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        # 修正索引值到: [0, self.word_count - 1]
        start = min(max(idx, 0), self.word_count - self.num_chars - 2)

        x = self.corpus_idx[start: start + self.num_chars]
        y = self.corpus_idx[start + 1: start + 1 + self.num_chars]

        return torch.tensor(x), torch.tensor(y)


def test02():
    _, _, _, corpus_idx = build_vocab()
    lyrics = LyricsDataset(corpus_idx, 5)
    lyrics_dataloader = DataLoader(lyrics, shuffle=False, batch_size=1)

    for x, y in lyrics_dataloader:
        print('x:', x)
        print('y:', y)
        break


def test001():
    index_to_word, word_to_index, word_count, corpus_idx = build_vocab()
    print(word_count)
    print(index_to_word)
    print(word_to_index)
    print(corpus_idx)


# 1. RNN 送入单个数据
def test01():
    # 输入数据维度 128, 输出维度 256
    rnn = nn.RNN(input_size=128, hidden_size=256)

    # 第一个数字: 表示句子长度
    # 第二个数字: 批量个数
    # 第三个数字: 表示数据维度
    inputs = torch.randn(1, 1, 128)
    hn = torch.zeros(1, 1, 256)
    print(f'inputs:\n{inputs},\nhn:\n{hn}')

    output, hn = rnn(inputs, hn)
    # print(f'output:\n{output},hn:\n{hn}')
    print(output.shape)
    print(hn.shape)


# 2. RNN层送入批量数据
def test02():
    # 输入数据维度 128, 输出维度 256
    rnn = nn.RNN(input_size=128, hidden_size=256)

    # 第一个数字: 表示句子长度
    # 第二个数字: 批量个数
    # 第三个数字: 表示数据维度
    inputs = torch.randn(1, 32, 128)
    hn = torch.zeros(1, 32, 256)

    output, hn = rnn(inputs, hn)
    print(output.shape)
    print(hn.shape)


if __name__ == '__main__':
    # test01()
    # test02()
    test001()
