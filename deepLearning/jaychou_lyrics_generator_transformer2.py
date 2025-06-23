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
import jieba
import jieba.posseg as pseg
from collections import Counter, defaultdict

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


# 数据预处理函数
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

    # 保留重要的断句符号
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：“”‘’（）《》【】{}()\[\]<>—-—,.!?;:\'\"`~@#$%^&*()_+=\-|\\\/*]'
    text = re.sub(pattern, '', text)

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

    # 确保包含特殊标记
    special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']

    # 将特殊标记添加到词汇表开头
    for token in reversed(special_tokens):
        if token not in unique_tokens:
            unique_tokens.insert(0, token)

    # 构建映射
    token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    idx_to_token = {i: token for i, token in enumerate(unique_tokens)}
    vocab_size = len(unique_tokens)

    print(f"词汇表大小: {vocab_size} 个唯一token，包含特殊标记")

    return tokens, token_to_idx, idx_to_token, vocab_size


# 歌词数据集类
class LyricsDataset(Dataset):
    def __init__(self, tokens, seq_length, token_to_idx):
        self.tokens = tokens
        self.seq_length = seq_length
        self.token_to_idx = token_to_idx
        self.total_sequences = len(tokens) - seq_length - 1
        self.indices = np.arange(self.total_sequences)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_length

        input_tokens = self.tokens[start_idx:end_idx]
        target_tokens = self.tokens[start_idx + 1:end_idx + 1]

        input_tensor = torch.tensor(
            [self.token_to_idx.get(token, self.token_to_idx['<unk>']) for token in input_tokens], dtype=torch.long)
        target_tensor = torch.tensor(
            [self.token_to_idx.get(token, self.token_to_idx['<unk>']) for token in target_tokens], dtype=torch.long)

        return input_tensor, target_tensor


# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(embed_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)

        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


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


# 模型加载助手函数
def load_model_state(model, checkpoint_path, strict=True):
    logger.info(f"尝试加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        checkpoint_state = checkpoint['model_state_dict']
    else:
        checkpoint_state = checkpoint

    model_state = model.state_dict()

    matched_params = {}
    unmatched_params = []

    for name, param in checkpoint_state.items():
        if name in model_state and param.shape == model_state[name].shape:
            matched_params[name] = param
        else:
            unmatched_params.append(name)

    logger.info(f"匹配的参数: {len(matched_params)}/{len(model_state)}")
    if unmatched_params:
        logger.warning(f"未匹配的参数: {len(unmatched_params)}")
        logger.debug(f"未匹配参数列表: {unmatched_params}")

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
    os.makedirs(save_path, exist_ok=True)

    model.train()
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, (inputs, targets) in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_description(
                f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Avg Loss: {epoch_loss / (i + 1):.4f}')

        progress_bar.close()

        avg_loss = epoch_loss / len(data_loader)
        perplexity = math.exp(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time

        logger.info(
            f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Time: {epoch_time:.2f}s, LR: {current_lr:.8f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{save_path}best_model.pt')
            logger.info(f"Epoch {epoch + 1}: 保存最佳模型，损失={avg_loss:.4f}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, f'{save_path}checkpoint.pt')

        if current_lr < lr_threshold:
            logger.info(f"Epoch {epoch + 1}: 学习率 {current_lr:.8f} 低于阈值 {lr_threshold}, 提前终止训练")
            break

        if avg_loss >= best_loss:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f"Epoch {epoch + 1}: 触发早停，最佳损失={best_loss:.4f}")
                break
        else:
            early_stop_counter = 0

    model.load_state_dict(torch.load(f'{save_path}best_model.pt'))
    return model


# 文本生成函数
def generate_text(model, token_to_idx, idx_to_token, seed_text, max_length=200, temperature=0.8, use_tokenizer=True):
    model.eval()

    if use_tokenizer:
        seed_tokens = jieba.lcut(seed_text)
    else:
        seed_tokens = list(seed_text)

    seed_tokens = [token for token in seed_tokens if token in token_to_idx]
    if not seed_tokens:
        print("种子文本中的所有词都不在词汇表中，使用随机种子")
        seed_tokens = [random.choice(list(token_to_idx.keys()))]

    input_idx = [token_to_idx[token] for token in seed_tokens]
    generated_text = seed_text

    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor([input_idx], dtype=torch.long).to(device)
            output = model(input_tensor)
            output = output[:, -1, :] / temperature
            probs = torch.softmax(output, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_token = idx_to_token[next_idx]
            generated_text += next_token
            input_idx = input_idx[1:] + [next_idx]

    return generated_text


# 基于规则的中文断句器 - 不依赖paddle
class ChineseSentenceSegmenter:
    def __init__(self):
        # 句子结束符
        self.sentence_enders = {'。', '！', '？', '!', '?', '…', '；', ';', '」', '”', '’', '》', '】', '）', ')'}

        # 分句符
        self.clause_breakers = {'，', '、', ',', '：', ':', '（', '(', '「', '“', '‘', '《', '【', '['}

        # 常见的句子起始词
        self.sentence_starters = {'但是', '然而', '所以', '因此', '而且', '不过', '因为', '虽然', '即使', '尽管',
                                  '如果', '只要', '只有', '于是', '那么', '可是', '还是', '或者', '要么', '首先',
                                  '其次', '最后', '另外', '此外'}

        # 常见的句子结束词
        self.sentence_end_words = {'了', '啊', '呀', '呢', '吧', '吗', '嘛', '啦', '哟', '哎', '罢了', '而已', '啊',
                                   '呀', '哇', '哪'}

        # 常见的分句起始词
        self.clause_starters = {'和', '与', '或', '还是', '并且', '同时', '接着', '然后', '此外', '另外', '以及',
                                '包括', '比如', '例如'}

        # 最小句子长度
        self.min_sentence_length = 5

        # 最大句子长度
        self.max_sentence_length = 100

    def segment(self, text):
        """将文本分割成句子"""
        if not text:
            return []

        sentences = []
        current_sentence = ""
        length = len(text)

        for i, char in enumerate(text):
            current_sentence += char

            # 检查是否是句子结束符
            is_end_char = char in self.sentence_enders

            # 检查是否达到最大句子长度
            is_too_long = len(current_sentence) >= self.max_sentence_length

            # 检查下一个字符是否是句子起始词
            next_is_starter = False
            if i < length - 1:
                for starter in self.sentence_starters:
                    if text[i + 1:].startswith(starter):
                        next_is_starter = True
                        break

            # 检查当前字符是否是句子结束词
            is_end_word = False
            for end_word in self.sentence_end_words:
                if current_sentence.endswith(end_word):
                    is_end_word = True
                    break

            # 综合判断是否应该断句
            should_break = (
                    (is_end_char and len(current_sentence) >= self.min_sentence_length) or
                    (is_too_long and (is_end_char or is_end_word)) or
                    (next_is_starter and len(current_sentence) >= self.min_sentence_length)
            )

            if should_break:
                sentences.append(current_sentence.strip())
                current_sentence = ""

        # 处理剩余文本
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return sentences


# 智能断句器 - 不依赖paddle
class IntelligentPunctuator:
    def __init__(self):
        self.segmenter = ChineseSentenceSegmenter()
        self.default_threshold = 0.6  # 断句阈值
        self.min_sentence_length = 5  # 最小句子长度

    def punctuate(self, text, threshold=None, min_length=None):
        """为文本添加标点和换行"""
        if threshold is None:
            threshold = self.default_threshold
        if min_length is None:
            min_length = self.min_sentence_length

        # 分割文本为句子
        sentences = self.segmenter.segment(text)

        # 重新组合文本，添加换行和标点
        punctuated_text = ""
        for i, sentence in enumerate(sentences):
            # 移除句子末尾可能存在的多余标点
            while sentence and sentence[-1] in {'，', '、', ',', ';', '；', '：', ':'}:
                sentence = sentence[:-1]

            # 添加正确的句末标点
            if not sentence:
                continue

            if sentence[-1] not in {'。', '！', '？', '!', '?', '…', ';', '；'}:
                sentence += '。'

            punctuated_text += sentence

            # 为歌词添加换行（每句后换行，空行分隔段落）
            if i < len(sentences) - 1:
                # 如果下一句以段落起始词开头，添加空行
                if any(sentences[i + 1].startswith(starter) for starter in self.segmenter.sentence_starters):
                    punctuated_text += '\n\n'
                else:
                    punctuated_text += '\n'

        return punctuated_text


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
        'epochs': 1,
        'save_path': 'models/',
        'seed_text': "窗外的麻雀",
        'max_generate_length': 300,
        'temperature': 0.8,
        'resume_training': False,
        'use_tokenizer': True,
        # 断句参数
        'break_threshold': 0.6,  # 断句阈值
        'min_sentence_length': 5,  # 最小句子长度
    }

    # 加载数据
    tokens, token_to_idx, idx_to_token, vocab_size = preprocess_data(config['file_path'], config['use_tokenizer'])

    # 创建模型
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

    # 创建断句器
    punctuator = IntelligentPunctuator()

    # 创建数据集和数据加载器
    dataset = LyricsDataset(tokens, config['seq_length'], token_to_idx)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

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

    if os.path.exists(checkpoint_path) and config['resume_training']:
        model = load_model_state(model, checkpoint_path, strict=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"从 epoch {start_epoch} 恢复训练")
        logger.info(f"之前的最佳损失: {checkpoint['loss']:.4f}")

    if config['resume_training'] or not os.path.exists(f"{config['save_path']}best_model.pt"):
        logger.info(f"开始训练，总轮数: {config['epochs']}")
        model = train(model, data_loader, criterion, optimizer, scheduler,
                      config['epochs'], device, config['save_path'], start_epoch=start_epoch)

    # 生成文本
    logger.info("开始生成文本...")
    generated_text = generate_text(
        model, token_to_idx, idx_to_token,
        config['seed_text'], config['max_generate_length'],
        config['temperature'], config['use_tokenizer']
    )

    # 使用智能断句器优化断句
    logger.info("应用智能断句...")
    punctuated_text = punctuator.punctuate(
        generated_text,
        threshold=config['break_threshold'],
        min_length=config['min_sentence_length']
    )

    # 打印歌词
    print("\n===== 生成的歌词 =====")
    print(punctuated_text)
    print("=====================")


if __name__ == "__main__":
    # 启用多进程
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')

    main()