import torch
import torch.nn as nn
import numpy as np
import random
import time
import math
import os
import zipfile
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LyricsDataset(Dataset):
    def __init__(self, text, seq_length, char_to_idx, idx_to_char, vocab_size):
        self.text = text
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = vocab_size

        # 计算可用的序列数量
        self.num_sequences = len(text) - seq_length - 1

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # 获取输入序列和目标序列
        input_seq = self.text[idx:idx + self.seq_length]
        target_seq = self.text[idx + 1:idx + self.seq_length + 1]

        # 将字符转换为索引
        input_idx = [self.char_to_idx[char] for char in input_seq]
        target_idx = [self.char_to_idx[char] for char in target_seq]

        # 转换为PyTorch张量
        input_tensor = torch.tensor(input_idx, dtype=torch.long)
        target_tensor = torch.tensor(target_idx, dtype=torch.long)

        return input_tensor, target_tensor


# 使用更高效的LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义网络层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # 嵌入层
        embedded = self.embedding(x)

        # LSTM层
        output, hidden = self.lstm(embedded, hidden)

        # 全连接层
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):
        # 初始化LSTM隐藏状态和细胞状态
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))


# 使用tqdm显示进度条，优化训练循环
def train(model, data_loader, criterion, optimizer, scheduler, epochs, device, save_path='models/', patience=3,
          start_epoch=0):
    # 创建保存模型的目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.train()
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        start_time = time.time()
        hidden = None

        # 使用tqdm显示进度条
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # 为每个批次创建新的隐藏状态
            if hidden is None or hidden[0].size(1) != batch_size:
                hidden = model.init_hidden(batch_size)
            else:
                # 分离隐藏状态，防止梯度累积
                hidden = (hidden[0].detach(), hidden[1].detach())

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

            # 更新进度条
            progress_bar.set_description(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

        # 计算平均损失
        avg_loss = total_loss / len(data_loader)

        # 学习率调度
        scheduler.step(avg_loss)

        # 打印训练信息
        elapsed = time.time() - start_time
        print(f'Epoch: {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | '
              f'Perplexity: {math.exp(avg_loss):.2f} | Time: {elapsed:.2f}s')

        # 保存最佳模型和最新模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{save_path}best_model.pt')
            print(f'Best model saved with loss: {avg_loss:.4f}')

        # 只保存最新的检查点，覆盖之前的
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, f'{save_path}checkpoint.pt')

        # 早停检查
        if avg_loss >= best_loss:
            early_stop_counter += 1
            print(f'Early stopping counter: {early_stop_counter}/{patience}')
            if early_stop_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
        else:
            early_stop_counter = 0

    # 加载最佳模型
    model.load_state_dict(torch.load(f'{save_path}best_model.pt'))
    return model


def generate_text(model, char_to_idx, idx_to_char, seed_text, max_length=200, temperature=1.0):
    model.eval()

    # 将种子文本转换为索引
    input_idx = [char_to_idx[char] for char in seed_text]
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

            # 采样下一个字符索引
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx_to_char[next_idx]

            # 添加到生成的文本中
            generated_text += next_char

            # 准备下一个输入
            input_idx = [next_idx]
            input_tensor = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0).to(device)

    return generated_text


# 数据预处理函数 - 更新为处理ZIP文件
def preprocess_data(file_path):
    text = ""

    # 检查是否为ZIP文件
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

    # 创建字符到索引和索引到字符的映射
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    print(f"语料库大小: {len(text)}")
    print(f"词汇表大小: {vocab_size}")

    return text, char_to_idx, idx_to_char, vocab_size


# 主函数
def main():
    # 参数配置
    config = {
        'file_path': 'jaychou_lyrics.txt.zip',  # 请替换为实际的歌词ZIP文件路径
        'seq_length': 100,
        'batch_size': 128,
        'embed_size': 256,
        'hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.5,
        'learning_rate': 0.002,
        'epochs': 20,
        'save_path': 'models/',
        'seed_text': "窗外的麻雀",  # 可以替换为你喜欢的任意歌词开头
        'max_generate_length': 300,
        'temperature': 0.8,
        'resume_training': True  # 是否从检查点恢复训练
    }

    # 数据预处理
    text, char_to_idx, idx_to_char, vocab_size = preprocess_data(config['file_path'])

    # 创建数据集和数据加载器
    dataset = LyricsDataset(text, config['seq_length'], char_to_idx, idx_to_char, vocab_size)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    # 初始化模型
    model = LSTMModel(vocab_size, config['embed_size'], config['hidden_size'], config['num_layers'],
                      config['dropout']).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 添加学习率调度
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # 检查是否有检查点文件
    start_epoch = 0
    if config['resume_training'] and os.path.exists(f"{config['save_path']}checkpoint.pt"):
        checkpoint = torch.load(f"{config['save_path']}checkpoint.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"从 epoch {start_epoch} 恢复训练")
        print(f"之前的最佳损失: {checkpoint['loss']:.4f}")
    else:
        print("从头开始训练新模型")

    # 训练模型
    model = train(model, data_loader, criterion, optimizer, scheduler,
                  config['epochs'], device, config['save_path'], start_epoch=start_epoch)

    # 生成歌词
    generated_lyrics = generate_text(model, char_to_idx, idx_to_char,
                                     config['seed_text'], config['max_generate_length'],
                                     config['temperature'])

    print("\n生成的歌词:")
    print(generated_lyrics)


if __name__ == "__main__":
    main()