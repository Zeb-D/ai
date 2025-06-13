import torch
import torch.nn as nn
import jieba

if __name__ == '__main__':

    text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'

    # 1. 文本分词
    words = jieba.lcut(text)
    print('文本分词:', words)

    # 2. 构建词表
    index_to_word = {}
    word_to_index = {}

    # 分词去重并保留原来的顺序
    unique_words = list(set(words))
    for idx, word in enumerate(unique_words):
        index_to_word[idx] = word
        word_to_index[word] = idx

    print(f'word_to_index:\n{word_to_index}')
    # 3. 构建词嵌入层
    # num_embeddings: 表示词表词的总数量
    # embedding_dim: 表示词嵌入的维度
    embed = nn.Embedding(num_embeddings=len(index_to_word), embedding_dim=4)

    # 4. 文本转换为词向量表示
    print('-' * 82)
    for word in words:
        # 获得词对应的索引
        idx = word_to_index[word]
        # 获得词嵌入向量
        t = torch.tensor(idx)
        word_vec = embed(t)
        print('t', t, '%3s\t' % word, word_vec)
