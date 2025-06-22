import re

import torch
import torch.nn as nn
import numpy as np
import random
import time
import math
import os
import zipfile
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import logging
import multiprocessing as mp
import jieba  # 添加分词器依赖

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cpu")
# 优先使用 CUDA，其次使用 MPS，最后使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS-(Metal Performance Shaders) for GPU acceleration")
# 检查 DirectML (Windows 上的 DirectX 12 加速)
elif os.name == 'nt':  # Windows 系统
    try:
        import torch_directml  # 在windows 执行 pip install torch-directml

        dml_device = torch_directml.device()
        device = dml_device
        print(f"Using DirectML for GPU acceleration on device: {torch_directml.device_name(dml_device)}")
    except ImportError:
        print("DirectML not installed. Using CPU.")
        print("Install DirectML with: pip install torch-directml")


# 数据预处理函数 - 优化分词处理
def preprocess_data(file_path, use_tokenizer=True):
    print(f"开始处理数据文件: {file_path}")
    text = ""

    # 检查是否为ZIP文件
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # 获取所有文件名
            file_names = zip_ref.namelist()
            logger.info(f"ZIP文件包含 {len(file_names)} 个文件")

            # 使用进度条读取文本文件内容
            with tqdm(file_names, desc="读取文件", unit="file") as pbar:
                for name in pbar:
                    if name.endswith('.txt'):
                        pbar.set_postfix({"当前文件": name})
                        with zip_ref.open(name) as f:
                            try:
                                content = f.read().decode('utf-8')
                            except UnicodeDecodeError:
                                content = f.read().decode('gbk')
                            text += content

    print(f"去除特殊字符前文本长度: {text}")
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：“”‘’（）《》【】{}()\[\]<>—-—,.!?;:\'\"`~@#$%^&*()_+=\-|\\\/*]'
    text = re.sub(pattern, '', text)
    # 规范化空格和换行
    # text = re.sub(r'\s+', ' ', text)  # 多个空格合并为一个
    # text = re.sub(r'[ \t]+$', '', text, flags=re.M)  # 去除行尾空格
    print(f"去除特殊字符后文本长度: {len(text)}")

    # 应用分词器
    if use_tokenizer:
        logger.info("使用分词器处理文本...")
        words = jieba.lcut(text)
        # 过滤空字符串
        tokens = [word for word in words if word.strip()]
        logger.info(f"分词后总词数: {len(tokens)}")
    else:
        # 未使用分词器，按字符处理
        tokens = [char for char in text if char.strip()]
        logger.info(f"总字符数: {len(tokens)}")

    # 创建词到索引和索引到词的映射
    unique_tokens = sorted(list(set(tokens)))

    # 确保包含特殊标记（小写和大写形式）
    special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>',
                      '<PAD>', '<UNK>', '<BOS>', '<EOS>',
                      '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    # 将特殊标记添加到词汇表开头
    for token in reversed(special_tokens):  # 逆序添加确保顺序正确
        if token not in unique_tokens:
            unique_tokens.insert(0, token)

    # 构建映射
    token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    idx_to_token = {i: token for i, token in enumerate(unique_tokens)}
    vocab_size = len(unique_tokens)

    print(f"词汇表大小: {vocab_size} 个唯一token，包含特殊标记")

    return tokens, token_to_idx, idx_to_token, vocab_size


# 优化的歌词数据集类 - 处理分词后的词列表
class LyricsDataset(Dataset):
    def __init__(self, tokens, seq_length, token_to_idx):
        self.tokens = tokens  # 现在是词列表而非原始文本
        self.seq_length = seq_length
        self.token_to_idx = token_to_idx
        self.total_sequences = len(tokens) - seq_length - 1

        # 预计算所有序列的起始索引，提高访问速度
        self.indices = np.arange(self.total_sequences)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        # 直接使用预计算的索引
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_length

        # 获取输入序列和目标序列
        input_tokens = self.tokens[start_idx:end_idx]
        target_tokens = self.tokens[start_idx + 1:end_idx + 1]

        # 将词转换为索引并转为张量
        input_tensor = torch.tensor([self.token_to_idx[token] for token in input_tokens], dtype=torch.long)
        target_tensor = torch.tensor([self.token_to_idx[token] for token in target_tokens], dtype=torch.long)

        return input_tensor, target_tensor


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # 嵌入层和位置编码
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出层
        self.decoder = nn.Linear(embed_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)

        # Transformer编码
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)
        output = self.transformer_encoder(src, src_mask)

        # 解码为词汇表大小
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# 模型加载助手函数
def load_model_state(model, checkpoint_path, strict=True):
    logger.info(f"尝试加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 获取检查点中的模型状态
    if 'model_state_dict' in checkpoint:
        checkpoint_state = checkpoint['model_state_dict']
    else:
        checkpoint_state = checkpoint

    # 检查模型类型是否匹配
    model_state = model.state_dict()

    # 查找匹配的参数
    matched_params = {}
    unmatched_params = []

    for name, param in checkpoint_state.items():
        if name in model_state and param.shape == model_state[name].shape:
            matched_params[name] = param
        else:
            unmatched_params.append(name)

    # 显示匹配情况
    logger.info(f"匹配的参数: {len(matched_params)}/{len(model_state)}")
    if unmatched_params:
        logger.warning(f"未匹配的参数: {len(unmatched_params)}")
        logger.debug(f"未匹配参数列表: {unmatched_params}")

    # 加载匹配的参数
    if matched_params:
        model_state.update(matched_params)
        model.load_state_dict(model_state, strict=False)
        logger.info("成功加载匹配的模型参数")
    else:
        logger.warning("未找到匹配的模型参数，使用随机初始化参数")

    return model


# 训练函数
def train(model, data_loader, criterion, optimizer, scheduler, epochs, device, save_path='models/', patience=3,
          start_epoch=0, lr_threshold=1e-6):
    # 创建保存模型的目录
    os.makedirs(save_path, exist_ok=True)

    model.train()
    best_loss = float('inf')
    early_stop_counter = 0

    # 指标记录
    metrics = {'best_loss': best_loss, 'current_loss': None, 'perplexity': None, 'early_stop': 0}

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0

        # 每个epoch的批次进度条
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, (inputs, targets) in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 梯度累积步数 - 模拟大批次训练
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # 反向传播
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # 优化器更新
            optimizer.step()

            # 学习率调度器更新
            scheduler.step()

            epoch_loss += loss.item()

            # 更新进度条
            progress_bar.set_description(
                f'Epoch-b {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, cur_loss: {epoch_loss:.4f}')

        # 关闭批次进度条
        progress_bar.close()

        # 计算平均损失
        avg_loss = epoch_loss / len(data_loader)
        perplexity = math.exp(avg_loss)

        # 更新指标
        metrics['current_loss'] = avg_loss
        metrics['perplexity'] = perplexity

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 计算当前epoch耗时
        epoch_time = time.time() - epoch_start_time

        # 保存最佳模型和最新模型
        if avg_loss < metrics['best_loss']:
            metrics['best_loss'] = avg_loss
            torch.save(model.state_dict(), f'{save_path}best_model.pt')
            logger.info(f"Epoch {epoch + 1}: 保存最佳模型，损失={avg_loss:.4f}")

        # 保存检查点
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, f'{save_path}checkpoint.pt')

        # 学习率阈值检查
        if current_lr < lr_threshold:
            logger.info(f"Epoch {epoch + 1}: 学习率 {current_lr:.8f} 低于阈值 {lr_threshold}, 提前终止训练")
            break

        # 早停检查
        if avg_loss >= metrics['best_loss']:
            early_stop_counter += 1
            metrics['early_stop'] = early_stop_counter
            if early_stop_counter >= patience:
                logger.info(f"Epoch {epoch + 1}: 触发早停，最佳损失={metrics['best_loss']:.4f}")
                break
        else:
            early_stop_counter = 0

    # 加载最佳模型
    model.load_state_dict(torch.load(f'{save_path}best_model.pt'))
    return model


# 文本生成函数 - 适应分词器
def generate_text(model, token_to_idx, idx_to_token, seed_text, max_length=200, temperature=0.8, use_tokenizer=True,
                  beam_width=1, unk_penalty=2.0):
    """生成歌词文本"""
    model.eval()

    # 处理种子文本
    if use_tokenizer:
        seed_tokens = jieba.lcut(seed_text)
    else:
        seed_tokens = list(seed_text)

    # 过滤不在词汇表中的词
    seed_tokens = [token for token in seed_tokens if token in token_to_idx]
    if not seed_tokens:
        print("种子文本中的所有词都不在词汇表中，使用随机种子")
        seed_tokens = [random.choice(list(token_to_idx.keys()))]

    input_idx = [token_to_idx[token] for token in seed_tokens]

    # 获取<unk> token的索引
    unk_idx = token_to_idx.get('<unk>', None)

    # 文本生成
    generated_text = seed_text

    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor([input_idx], dtype=torch.long).to(device)
            output = model(input_tensor)

            # 应用温度参数调整预测分布
            output = output[:, -1, :] / temperature

            # 惩罚生成<unk> token
            if unk_idx is not None:
                output[0, unk_idx] -= unk_penalty

            # 采样下一个词
            probs = torch.softmax(output, dim=1)

            if beam_width > 1:
                # 使用beam search
                next_indices = torch.topk(probs, beam_width, dim=1)[1][0].tolist()

                # 选择概率最高的词
                next_idx = next_indices[0]
            else:
                # 采样
                next_idx = torch.multinomial(probs, num_samples=1).item()

            # 添加到生成的索引中
            generated_text += idx_to_token[next_idx]
            input_idx = input_idx[1:] + [next_idx]  # 滑动窗口

    # 过滤特殊token
    filtered_tokens = []
    for token in generated_text:
        if token not in ['<pad>', '<bos>', '<eos>']:
            filtered_tokens.append(token)

    # 合并为文本
    if use_tokenizer:
        generated_text = ''.join(filtered_tokens)
        # 简单的后处理，提高文本可读性
        generated_text = re.sub(r'([^\u4e00-\u9fa5])([\u4e00-\u9fa5])', r'\1 \2', generated_text)
        generated_text = re.sub(r'([\u4e00-\u9fa5])([^\u4e00-\u9fa5])', r'\1 \2', generated_text)
    else:
        generated_text = ''.join(filtered_tokens)

    return generated_text


# 主函数
def main():
    # 参数配置
    config = {
        'file_path': 'jaychou_lyrics.txt.zip',  # 请替换为实际的歌词ZIP文件路径
        'seq_length': 100,
        'batch_size': 64,
        'embed_size': 256,
        'hidden_size': 512,
        'num_layers': 2,
        'num_heads': 8,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 15,
        'save_path': 'models/',
        'seed_text': "窗外的麻雀",
        'max_generate_length': 300,
        'temperature': 0.8,
        'resume_training': False,
        'num_workers': min(8, mp.cpu_count()),
        'pin_memory': True,
        'strict_loading': False,
        'lr_threshold': 1e-6,  # 学习率低于此值时终止训练
        'use_tokenizer': True,  # 是否使用分词器
    }

    tokens, token_to_idx, idx_to_token, vocab_size = preprocess_data(config['file_path'], config['use_tokenizer'])

    # 创建数据集和数据加载器
    dataset = LyricsDataset(tokens, config['seq_length'], token_to_idx)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True,
        prefetch_factor=2
    )

    # 初始化模型
    model = TransformerModel(
        vocab_size=vocab_size,
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)

    if device.type == 'cuda':
        model = torch.nn.parallel.DataParallel(model)
        logger.info("启用数据并行训练")

    # 设置优化器和调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        steps_per_epoch=len(data_loader),
        epochs=config['epochs'],
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # 检查并加载检查点
    start_epoch = 0
    checkpoint_path = f"{config['save_path']}checkpoint.pt"

    resume_training = config['resume_training']
    if os.path.exists(checkpoint_path):
        model = load_model_state(model, checkpoint_path, strict=config['strict_loading'])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"从 epoch {start_epoch} 恢复训练")
        logger.info(f"之前的最佳损失: {checkpoint['loss']:.4f}")
    else:
        resume_training = True

    if resume_training:
        # 训练模型
        logger.info(f"开始训练，总轮数: {config['epochs']}, 学习率阈值: {config['lr_threshold']}")
        model = train(model, data_loader, criterion, optimizer, scheduler,
                      config['epochs'], device, config['save_path'], start_epoch=start_epoch,
                      lr_threshold=config['lr_threshold'])

    # 生成歌词
    generated_lyrics = generate_text(model, token_to_idx, idx_to_token,
                                     config['seed_text'], config['max_generate_length'],
                                     config['temperature'], config['use_tokenizer'])

    # 打印歌词
    print("\n===== 生成的歌词 =====")
    print(generated_lyrics)
    print("=====================")


if __name__ == "__main__":
    # 启用多进程
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')

    main()
