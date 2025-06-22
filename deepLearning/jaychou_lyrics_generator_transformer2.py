import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import math
import os
import zipfile
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import logging
import multiprocessing as mp
import jieba
from collections import Counter
import re
import copy

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 设置随机种子，保证结果可复现
def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seeds()

# 选择设备
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS-2(Metal Performance Shaders) for GPU acceleration")


# 增强的分词处理函数
def enhanced_tokenize(text, use_jieba=True, add_word_list=None, remove_stopwords=True):
    if add_word_list:
        for word in add_word_list:
            jieba.add_word(word)

    text = re.sub(r'\[.*?\]', '', text)  # 移除方括号内的内容，如[Verse 1]
    eng_words = re.findall(r'[a-zA-Z]+', text)
    for word in eng_words:
        jieba.add_word(word)

    if use_jieba:
        tokens = jieba.lcut(text)
    else:
        tokens = list(text)

    if remove_stopwords:
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很',
                     '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        tokens = [token for token in tokens if token not in stopwords]

    return tokens


# 数据预处理函数
def preprocess_data(file_path, use_tokenizer=True, add_word_list=None, min_freq=2, unk_threshold=0.1):
    logger.info(f"开始处理数据文件: {file_path}")
    text = ""

    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            logger.info(f"ZIP文件包含 {len(file_names)} 个文件")

            for name in file_names:
                if name.endswith('.txt'):
                    with zip_ref.open(name) as f:
                        try:
                            content = f.read().decode('utf-8')
                        except UnicodeDecodeError:
                            content = f.read().decode('gbk')
                        text += content

    if use_tokenizer:
        logger.info("使用增强分词器处理文本...")
        tokens = enhanced_tokenize(text, True, add_word_list)
        logger.info(f"分词后总词数: {len(tokens)}")
    else:
        tokens = list(text)
        logger.info(f"总字符数: {len(tokens)}")

    token_freq = Counter(tokens)
    logger.info(f"唯一token数: {len(token_freq)}")

    low_freq_tokens = [token for token, freq in token_freq.items() if freq < min_freq]
    low_freq_ratio = len(low_freq_tokens) / len(token_freq) if len(token_freq) > 0 else 0

    if min_freq > 1 and low_freq_ratio < unk_threshold:
        filtered_tokens = [token for token in tokens if token_freq[token] >= min_freq]
        unique_tokens = ['<pad>', '<unk>', '<bos>', '<eos>'] + [token for token in sorted(set(filtered_tokens)) if
                                                                token not in ['<pad>', '<unk>', '<bos>', '<eos>']]
        logger.info(f"过滤低频词后唯一token数: {len(unique_tokens)}")
        logger.info(f"低频词过滤率: {low_freq_ratio:.2%}")
    else:
        logger.info(f"低频词比例过高 ({low_freq_ratio:.2%}), 不过滤低频词")
        unique_tokens = ['<pad>', '<unk>', '<bos>', '<eos>'] + [token for token in sorted(set(tokens)) if
                                                                token not in ['<pad>', '<unk>', '<bos>', '<eos>']]

    token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    idx_to_token = {i: token for i, token in enumerate(unique_tokens)}

    if min_freq > 1 and low_freq_ratio < unk_threshold:
        tokens = [token if token_freq[token] >= min_freq else '<unk>' for token in tokens]

    return tokens, token_to_idx, idx_to_token, len(unique_tokens)


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


# 位置编码类
class EnhancedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, use_relative=False):
        super().__init__()
        self.use_relative = use_relative

        if use_relative:
            self.rel_pos_bias = nn.Embedding(2 * max_len - 1, d_model)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

    def forward(self, x):
        if self.use_relative:
            return x
        else:
            return x + self.pe[:x.size(1), :]


# 带有相对位置编码的多头注意力层
class RelativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, rel_pos_embedding=None, attn_mask=None, key_padding_mask=None):
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v = self._in_projection_packed(query, key, value)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)

        if rel_pos_embedding is not None:
            batch_size, seq_len = query.size(1), query.size(0)
            rel_pos_embedding = rel_pos_embedding.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim)
            rel_pos_embedding = rel_pos_embedding.permute(0, 3, 1, 2, 4)
            q_for_rel = q.view(bsz, self.num_heads, seq_len, self.head_dim)
            rel_attn_bias = torch.einsum('bhld,bhlmd->bhlm', q_for_rel, rel_pos_embedding)
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, seq_len, seq_len) + rel_attn_bias
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, seq_len, seq_len)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            else:
                raise ValueError("attn_mask 维度错误")
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, seq_len, seq_len)
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, seq_len, -1)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, seq_len, -1)

        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_output_weights

    def _in_projection_packed(self, q, k, v):
        bsz, tgt_len, embed_dim = q.size()
        w = self.in_proj_weight
        b = self.in_proj_bias

        q, k, v = F.linear(q, w, b).chunk(3, dim=-1)
        return q, k, v


# 自定义Transformer编码器层，支持相对位置编码
class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = RelativeMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, src, rel_pos_embedding=None, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2 = src2.permute(1, 0, 2)
        src2, _ = self.self_attn(
            src2, src2, src2,
            rel_pos_embedding=rel_pos_embedding,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src2 = src2.permute(1, 0, 2)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


# 自定义支持相对位置编码的Transformer编码器
class RelativeTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, rel_pos_embedding=None, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, rel_pos_embedding=rel_pos_embedding, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


# 增强的Transformer模型
class EnhancedTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout=0.5, use_relative_pos=False):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = EnhancedPositionalEncoding(embed_size, use_relative=use_relative_pos)
        self.use_relative_pos = use_relative_pos

        if use_relative_pos:
            encoder_layer = RelativeTransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                activation='gelu'
            )
            self.transformer_encoder = RelativeTransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(embed_size)
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(embed_size, vocab_size)
        self.norm = nn.LayerNorm(embed_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        src = self.norm(src)

        if src_mask is None:
            batch_size, seq_len = src.size(0), src.size(1)
            src_mask = torch.triu(torch.ones(batch_size, seq_len, seq_len, device=src.device), diagonal=1).bool()

        rel_pos_embedding = None
        if self.use_relative_pos:
            batch_size, seq_len, embed_size = src.size()
            context_position = torch.arange(seq_len, dtype=torch.long, device=src.device).unsqueeze(1)
            memory_position = torch.arange(seq_len, dtype=torch.long, device=src.device).unsqueeze(0)
            relative_position = memory_position - context_position
            relative_position = relative_position + seq_len - 1
            rel_pos_embedding = self.pos_encoder.rel_pos_bias(relative_position)
            rel_pos_embedding = rel_pos_embedding.unsqueeze(0).expand(batch_size, -1, -1, -1)

        if self.use_relative_pos:
            output = self.transformer_encoder(src, rel_pos_embedding=rel_pos_embedding, mask=src_mask)
        else:
            output = self.transformer_encoder(src, src_mask)

        output = self.decoder(output)
        return output


# 模型加载函数
def load_model_state(model, checkpoint_path, strict=True):
    if not os.path.exists(checkpoint_path):
        logger.info(f"检查点不存在: {checkpoint_path}")
        return model, 0, float('inf')

    logger.info(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_state_dict = checkpoint['model_state_dict']
    if next(iter(model_state_dict.keys())).startswith('module.') and not hasattr(model, 'module'):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}

    model.load_state_dict(model_state_dict, strict=strict)

    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))

    return model, epoch, best_loss


# 训练函数
def train(model, data_loader, criterion, optimizer, scheduler, epochs, device, save_path='models/', start_epoch=0,
          best_loss=float('inf'), patience=3):
    os.makedirs(save_path, exist_ok=True)
    early_stop_count = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        batch_progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for inputs, targets in batch_progress:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            batch_progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})

        avg_loss = epoch_loss / len(data_loader)
        perplexity = math.exp(avg_loss)

        logger.info(f"Epoch {epoch + 1}/{epochs} 平均损失: {avg_loss:.4f}, 困惑度: {perplexity:.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, f"{save_path}best_model.pt")
            early_stop_count = 0
            logger.info(f"保存最佳模型到 {save_path}best_model.pt, 损失: {best_loss:.4f}")
        else:
            early_stop_count += 1
            logger.info(f"早停计数: {early_stop_count}/{patience}")
            if early_stop_count >= patience:
                logger.info(f"触发早停，最佳损失: {best_loss:.4f}")
                break

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }, f"{save_path}current_model.pt")

    model, _, _ = load_model_state(model, f"{save_path}best_model.pt")
    return model


# 文本生成函数
def generate_text(model, token_to_idx, idx_to_token, seed_text, max_length=200, temperature=0.8, use_tokenizer=True,
                  beam_width=1, unk_penalty=2.0):
    model.eval()

    if use_tokenizer:
        seed_tokens = enhanced_tokenize(seed_text, True, None, False)
    else:
        seed_tokens = list(seed_text)

    seed_tokens = [token for token in seed_tokens if token in token_to_idx]
    if not seed_tokens:
        logger.warning("种子文本中的所有词都不在词汇表中，使用随机种子")
        seed_tokens = [random.choice(list(token_to_idx.keys()))]

    input_idx = [token_to_idx[token] for token in seed_tokens]
    unk_idx = token_to_idx.get('<unk>', None)
    generated_indices = []

    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor([input_idx], dtype=torch.long).to(device)
            output = model(input_tensor)

            output = output[:, -1, :] / temperature

            if unk_idx is not None:
                output[0, unk_idx] -= unk_penalty

            probs = torch.softmax(output, dim=1)

            if beam_width > 1:
                next_indices = torch.topk(probs, beam_width, dim=1)[1][0].tolist()
                next_probs = torch.topk(probs, beam_width, dim=1)[0][0].tolist()
                next_idx = next_indices[0]
            else:
                next_idx = torch.multinomial(probs, num_samples=1).item()

            generated_indices.append(next_idx)
            input_idx = input_idx[1:] + [next_idx]

    generated_tokens = [idx_to_token[idx] for idx in generated_indices]
    filtered_tokens = []
    for token in generated_tokens:
        if token not in ['<pad>', '<bos>', '<eos>']:
            filtered_tokens.append(token)

    if use_tokenizer:
        generated_text = ''.join(filtered_tokens)
        generated_text = re.sub(r'([^\u4e00-\u9fa5])([\u4e00-\u9fa5])', r'\1 \2', generated_text)
        generated_text = re.sub(r'([\u4e00-\u9fa5])([^\u4e00-\u9fa5])', r'\1 \2', generated_text)
    else:
        generated_text = ''.join(filtered_tokens)

    return generated_text


# 主函数
def main():
    config = {
        'file_path': 'jaychou_lyrics.txt.zip',
        'seq_length': 100,
        'batch_size': 64,
        'embed_size': 256,
        'hidden_size': 512,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 15,
        'save_path': 'models/',
        'seed_text': "窗外的麻雀",
        'max_generate_length': 300,
        'temperature': 0.8,
        'resume_training': True,
        'num_workers': min(8, mp.cpu_count()),
        'pin_memory': True,
        'strict_loading': False,
        'lr_threshold': 1e-6,
        'use_tokenizer': True,
        'use_relative_pos': True,
        'beam_width': 3,
        'min_freq': 2,
        'unk_threshold': 0.1,
        'unk_penalty': 2.0,
        'custom_words': ['方文山', '周杰伦', '嘻哈', 'R&B', '旋律', '歌词', '悬崖', '爱情', '眼泪', '微笑', '回忆',
                         '故事', '烟火', '星空', '流浪']
    }

    logger.info("开始数据预处理...")
    tokens, token_to_idx, idx_to_token, vocab_size = preprocess_data(
        config['file_path'],
        use_tokenizer=config['use_tokenizer'],
        add_word_list=config['custom_words'],
        min_freq=config['min_freq'],
        unk_threshold=config['unk_threshold']
    )

    dataset = LyricsDataset(tokens, config['seq_length'], token_to_idx)
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True
    )

    logger.info("初始化模型...")
    model = EnhancedTransformerModel(
        vocab_size=vocab_size,
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        use_relative_pos=config['use_relative_pos']
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")

    criterion = nn.CrossEntropyLoss(ignore_index=token_to_idx['<pad>'])
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=config['epochs'] * len(data_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1000.0
    )

    start_epoch = 0
    best_loss = float('inf')

    if config['resume_training']:
        model, start_epoch, best_loss = load_model_state(
            model,
            f"{config['save_path']}current_model.pt",
            strict=config['strict_loading']
        )
        logger.info(f"从 epoch {start_epoch} 恢复训练，最佳损失: {best_loss:.4f}")

    logger.info("开始训练模型...")
    model = train(
        model,
        data_loader,
        criterion,
        optimizer,
        scheduler,
        config['epochs'],
        device,
        config['save_path'],
        start_epoch,
        best_loss
    )

    logger.info("开始生成歌词...")
    generated_lyrics = generate_text(
        model,
        token_to_idx,
        idx_to_token,
        config['seed_text'],
        config['max_generate_length'],
        config['temperature'],
        config['use_tokenizer'],
        config['beam_width'],
        config['unk_penalty']
    )

    print("\n===== 生成的歌词 =====")
    print(generated_lyrics)
    print("=====================")

    with open(f"{config['save_path']}generated_lyrics.txt", 'w', encoding='utf-8') as f:
        f.write(generated_lyrics)

    logger.info(f"歌词已保存到 {config['save_path']}generated_lyrics.txt")


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    main()
