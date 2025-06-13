import torch
import torch.nn as nn
import numpy as np
import random
import time
import math
import os
import zipfile
import jieba
from torch.utils.data import Dataset, DataLoader

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检查是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LyricsDataset(Dataset):
    def __init__(self, words, seq_length, word_to_idx, idx_to_word, vocab_size):
        self.words = words
        self.seq_length = seq_length
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.vocab_size = vocab_size

        # 计算可用的序列数量
        self.num_sequences = len(words) - seq_length - 1

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # 获取输入序列和目标序列
        input_seq = self.words[idx:idx + self.seq_length]
        target_seq = self.words[idx + 1:idx + self.seq_length + 1]

        # 将词转换为索引
        input_idx = [self.word_to_idx[word] for word in input_seq]
        target_idx = [self.word_to_idx[word] for word in target_seq]

        # 转换为 PyTorch 张量
        input_tensor = torch.tensor(input_idx, dtype=torch.long)
        target_tensor = torch.tensor(target_idx, dtype=torch.long)

        return input_tensor, target_tensor


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义网络层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # 嵌入层
        embedded = self.embedding(x)

        # RNN 层
        output, hidden = self.rnn(embedded, hidden)

        # 全连接层
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):
        # 初始化隐藏状态
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


def train(model, data_loader, criterion, optimizer, epochs, device, save_path='models/', start_epoch=0):
    # 创建保存模型的目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.train()
    total_loss = 0
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        # 初始化隐藏状态
        hidden = None

        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # 为每个批次创建新的隐藏状态
            if hidden is None or hidden.size(1) != batch_size:
                hidden = model.init_hidden(batch_size)
            else:
                # 只保留前 batch_size 个样本的隐藏状态
                hidden = hidden[:, :batch_size, :].detach()

            # 前向传播
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

            # 打印训练进度
            if i % 100 == 0 and i > 0:
                cur_loss = total_loss / 100
                elapsed = time.time() - start_time
                progress_str = f'Epoch: {epoch + 1}/{epochs} | Batch: {i}/{len(data_loader)} | ' \
                               f'Loss: {cur_loss:.4f} | Perplexity: {math.exp(cur_loss):.2f} | ' \
                               f'Time: {elapsed:.2f}s'
                print(progress_str, end='\r')
                total_loss = 0
                start_time = time.time()

        # 每个 epoch 保存一次模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f'{save_path}model_epoch_{epoch + 1}.pt')
        print(f'\nModel saved at {save_path}model_epoch_{epoch + 1}.pt')

    return model


def generate_text(model, word_to_idx, idx_to_word, seed_text, max_length=200, temperature=1.0):
    model.eval()

    # 将种子文本分词并转换为索引
    seed_words = jieba.lcut(seed_text)
    input_idx = [word_to_idx[word] for word in seed_words if word in word_to_idx]
    input_tensor = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0).to(device)

    # 初始化隐藏状态
    hidden = model.init_hidden(1)

    # 生成文本
    generated_text = seed_text

    with torch.no_grad():
        for i in range(max_length):
            # 前向传播
            output, hidden = model(input_tensor, hidden)

            # 应用温度参数调整预测分布
            output = output[:, -1, :] / temperature
            probs = torch.softmax(output, dim=1)

            # 采样下一个词索引
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_word = idx_to_word[next_idx]

            # 添加到生成的文本中
            generated_text += next_word

            # 准备下一个输入
            input_idx = [next_idx]
            input_tensor = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0).to(device)

    return generated_text


# 数据预处理函数 - 更新为处理 ZIP 文件
def preprocess_data(file_path):
    text = ""

    # 检查是否为 ZIP 文件
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # 获取所有文件名
            file_names = zip_ref.namelist()

            # 读取所有文本文件内容
            for name in file_names:
                if name.endswith('.txt'):
                    with zip_ref.open(name) as f:
                        # 尝试不同的编码方式读取文件
                        try:
                            content = f.read().decode('utf-8')
                        except UnicodeDecodeError:
                            content = f.read().decode('gbk')
                        text += content
    else:
        # 处理普通文本文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

    # 使用 jieba 分词
    words = jieba.lcut(text)

    # 创建词到索引和索引到词的映射
    unique_words = sorted(list(set(words)))
    word_to_idx = {word: i for i, word in enumerate(unique_words)}
    idx_to_word = {i: word for i, word in enumerate(unique_words)}
    vocab_size = len(unique_words)

    print(f"语料库词数: {len(words)}")
    print(f"词汇表大小: {vocab_size}")

    return words, word_to_idx, idx_to_word, vocab_size


# 主函数
def main():
    # 数据预处理
    file_path = 'jaychou_lyrics.txt.zip'  # 请替换为实际的歌词 ZIP 文件路径
    words, word_to_idx, idx_to_word, vocab_size = preprocess_data(file_path)

    # 创建数据集和数据加载器
    seq_length = 100
    batch_size = 64
    dataset = LyricsDataset(words, seq_length, word_to_idx, idx_to_word, vocab_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型参数
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    dropout = 0.5

    # 初始化模型
    model = RNNModel(vocab_size, embed_size, hidden_size, num_layers, dropout).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 检查是否有保存的模型
    save_path = 'models/'
    start_epoch = 0
    if os.path.exists(save_path):
        saved_models = [f for f in os.listdir(save_path) if f.startswith('model_epoch_')]
        if saved_models:
            latest_model = max(saved_models, key=lambda x: int(x.split('_')[2].split('.')[0]))
            checkpoint = torch.load(os.path.join(save_path, latest_model))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {start_epoch}")

    # 训练模型
    epochs = 30
    model = train(model, data_loader, criterion, optimizer, epochs, device, save_path, start_epoch)

    # 生成歌词
    seed_text = "窗外的鸡"  # 可以替换为你喜欢的任意歌词开头
    generated_lyrics = generate_text(model, word_to_idx, idx_to_word, seed_text,
                                     max_length=300, temperature=0.8)

    print("\n生成的歌词:")
    print(generated_lyrics)


if __name__ == "__main__":
    main()