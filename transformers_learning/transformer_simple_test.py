import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

torch.manual_seed(42)
device = config.device


###############################################################################
# 1. 缩放点积注意力（论文 3.2.1 节）
###############################################################################
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    计算缩放点积注意力。

    论文公式:  Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

    参数:
        query: [batch, heads, seq_len, d_k]  查询向量
        key:   [batch, heads, seq_len, d_k]  键向量
        value: [batch, heads, seq_len, d_k]  值向量
        mask:  [1, 1, seq_len, seq_len] 或 None
               mask 中值为 0 的位置将在 softmax 前被设为 -1e9
        dropout: nn.Dropout 或 None

    返回:
        out:  [batch, heads, seq_len, d_k] 注意力加权后的向量
        attn: [batch, heads, seq_len, seq_len] 注意力权重矩阵（可用于可视化）
    """
    d_k = query.size(-1)

    # 1. 计算 Q 与 K 的点积，除以 sqrt(d_k) 进行缩放
    #    scores 形状: [batch, heads, seq_len, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 应用 mask：将 mask == 0 的位置设为极小的负数，
    #    这样经过 softmax 后注意力权重会接近 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. softmax 得到注意力权重（每行和为 1）
    attn = F.softmax(scores, dim=-1)

    # 4. 训练时使用 dropout 防止过拟合
    if dropout is not None:
        attn = dropout(attn)

    # 5. 注意力权重乘以 value 得到输出
    out = torch.matmul(attn, value)  # [batch, heads, seq_len, d_k]
    return out, attn


###############################################################################
# 2. 多头注意力（论文 3.2.2 节）
###############################################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        """
        多头注意力将 Q、K、V 分别投影到 h 个子空间，然后并行计算注意力，
        最后将 h 个头的结果拼接并做一次线性变换。

        论文设置: d_model = 512, h = 8, d_k = d_v = d_model / h = 64
        """
        super().__init__()
        assert d_model % h == 0  # d_model 必须能被头数整除
        self.d_k = d_model // h  # 每个头的维度
        self.h = h  # 头的数量

        # 对 Q、K、V 分别做线性变换，输出维度仍为 d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 多头拼接后的输出线性变换
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        参数:
            query: [batch, seq_len, d_model]
            key:   [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask:  [1, 1, seq_len, seq_len] 或 None
        返回:
            [batch, seq_len, d_model]
        """
        batch = query.size(0)

        # 1. 线性映射: 将 query, key, value 分别投影到 d_model 维空间
        Q = self.w_q(query)  # [batch, seq_len, d_model]
        K = self.w_k(key)
        V = self.w_v(value)

        # 2. 拆分成多个头:
        #    先 view 成 [batch, seq_len, h, d_k]
        #    然后 transpose(1,2) 得到 [batch, h, seq_len, d_k]
        Q = Q.view(batch, -1, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch, -1, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch, -1, self.h, self.d_k).transpose(1, 2)

        # 3. 对每个头分别计算缩放点积注意力
        out, _ = scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        # out: [batch, h, seq_len, d_k]

        # 4. 拼接所有头:
        #    先转置回 [batch, seq_len, h, d_k]
        #    然后合并最后两个维度 -> [batch, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)

        # 5. 最后的线性变换
        return self.w_o(out)


###############################################################################
# 3. 位置前馈网络（论文 3.3 节）
###############################################################################
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        论文中的 FFN: 两层线性变换，中间用 ReLU 激活，内层维度 d_ff。
        原文: FFN(x) = max(0, xW1 + b1)W2 + b2
        论文设置: d_model = 512, d_ff = 2048
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 扩展到 d_ff 维
        self.linear2 = nn.Linear(d_ff, d_model)  # 压缩回 d_model 维
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


###############################################################################
# 4. 位置编码（论文 3.5 节）
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        由于 Transformer 没有循环或卷积结构，需要显式地告诉模型每个 token 的位置。
        论文使用正弦/余弦函数生成位置编码:
            PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        super().__init__()
        # 预先计算位置编码矩阵，形状 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]

        # 计算频率分母: div_term = exp( -log(10000) * (2i / d_model) )
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # register_buffer 使位置编码随模型移动到 GPU，但不参与反向传播更新
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # 将位置编码加到词嵌入上
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


###############################################################################
# 5. 编码器层（论文 3.1 节）
###############################################################################
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 两个 LayerNorm，分别用于自注意力输出和前馈网络输出
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, d_model]

        # 1. 自注意力 + 残差连接 + LayerNorm
        #    query, key, value 都来自 x
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. 前馈网络 + 残差连接 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


###############################################################################
# 6. 解码器层（论文 3.1 节）
###############################################################################
class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        # 解码器包含三个子层：
        #   a) Masked 自注意力（防止看到未来信息）
        #   b) Cross-Attention（query 来自 decoder，key/value 来自 encoder）
        #   c) 前馈网络
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        参数:
            x:       decoder 输入 [batch, tgt_len, d_model]
            memory:  encoder 输出 [batch, src_len, d_model]
            src_mask: [1, 1, tgt_len, src_len] 或 None  用于 cross-attention 屏蔽
            tgt_mask: [1, 1, tgt_len, tgt_len] 或 None  用于自注意力屏蔽未来
        """
        # 1. Masked 自注意力: query=key=value=x
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Cross-Attention: query=x (decoder), key=memory, value=memory (encoder)
        cross_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # 3. 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


###############################################################################
# 7. 完整 Transformer（论文 3.1 节）
###############################################################################
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=64, h=4, d_ff=128, N=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 词嵌入层：将 token id 映射为 d_model 维稠密向量
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # 堆叠 N 个 EncoderLayer（论文中 N=6）
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        # 堆叠 N 个 DecoderLayer
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])

        # 最终的线性层，将 d_model 维映射到目标词表大小，输出每个位置的预测 logits
        self.generator = nn.Linear(d_model, tgt_vocab)

    def encode(self, src, src_mask=None):
        # src: [batch, src_len]
        # 1. 词嵌入并乘以 sqrt(d_model)（论文提到这样做可以防止嵌入值过小）
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        # 2. 依次通过所有编码器层
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x  # 返回 encoder 输出（memory）

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        # tgt: [batch, tgt_len]
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器
        memory = self.encode(src, src_mask)
        # 解码器
        out = self.decode(tgt, memory, src_mask, tgt_mask)
        # 生成预测
        return self.generator(out)  # [batch, tgt_len, tgt_vocab]


###############################################################################
# 8. 生成 Look-ahead Mask（论文 3.1 节）
###############################################################################
def subsequent_mask(size):
    """
    返回一个下三角布尔矩阵，形状 [1, 1, size, size]。
    位置 (i, j) 为 True 表示第 i 个位置可以注意到第 j 个位置（j <= i）。
    用于防止解码器在预测时看到未来的 token。
    """
    mask = torch.tril(torch.ones((1, 1, size, size), dtype=torch.bool))
    return mask


###############################################################################
# 9. 数据准备与训练示例（长文本复制任务）
###############################################################################

# 构建字符级词汇表
chars = "abcdefghijklmnopqrstuvwxyz "  # 27 个字符（小写字母和空格）
vocab = {ch: i for i, ch in enumerate(chars)}
vocab['<sos>'] = len(vocab)  # 27
vocab['<eos>'] = len(vocab)  # 28
vocab_size = len(vocab)  # 29

char_to_idx = vocab
idx_to_char = {i: ch for ch, i in vocab.items()}


def text_to_tensor(text, add_sos=False, add_eos=False):
    """将字符串转换为 index 张量"""
    ids = [char_to_idx[ch] for ch in text]
    if add_sos:
        ids = [char_to_idx['<sos>']] + ids
    if add_eos:
        ids = ids + [char_to_idx['<eos>']]
    return torch.tensor(ids, dtype=torch.long)


def tensor_to_text(tensor):
    """将 index 张量转换为字符串（忽略特殊 token）"""
    chars = []
    for idx in tensor.tolist():
        if idx == char_to_idx['<sos>'] or idx == char_to_idx['<eos>']:
            continue
        chars.append(idx_to_char[idx])
    return ''.join(chars)


# 定义一个较长的固定句子用于测试
sentence = "the quick brown fox jumps over the lazy dog"
seq_len = len(sentence)  # 43 个字符

# 生成固定句子的训练/测试数据
src_sentence = text_to_tensor(sentence, add_sos=False, add_eos=False)  # [seq_len]

# 构建 decoder 输入: <sos> + 原句的前 seq_len-1 个字符，保持长度与 src 一致
sos_idx = char_to_idx['<sos>']
tgt_input_sentence = torch.cat([
    torch.tensor([sos_idx], dtype=torch.long),
    src_sentence[:-1]
], dim=0)  # [seq_len]

# 目标输出就是原句（长度 seq_len）
tgt_output_sentence = src_sentence.clone()


# 生成随机批次用于训练
def generate_batch(batch_size, seq_len):
    """
    生成随机字符序列，让模型学习复制能力。
    只使用前 27 个字符（索引 0~26），不包含特殊 token。
    """
    # 随机选择索引 0~26（对应字母和空格）
    src = torch.randint(0, 27, (batch_size, seq_len))
    tgt = src.clone()  # 目标就是源序列

    # decoder 输入: <sos> + target[:-1]
    tgt_input = torch.cat([
        torch.full((batch_size, 1), sos_idx, dtype=torch.long),
        tgt[:, :-1]
    ], dim=1)  # [batch, seq_len]

    return src, tgt_input, tgt


# 初始化模型（使用小规模配置以加快 CPU 训练）
model = Transformer(
    src_vocab=vocab_size,  # 29
    tgt_vocab=vocab_size,
    d_model=128,  # 嵌入维度（原始论文为 512，这里减小便于演示）
    h=8,  # 多头数量（128/8=16，每个头维度为16）
    d_ff=256,  # 前馈网络内层维度
    N=3,  # 编码器和解码器层数（论文中为6）
    dropout=0.1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 训练循环
num_steps = 1000
batch_size = 32

print("开始训练...")
for step in range(num_steps):
    model.train()

    # 生成随机批次
    src, tgt_input, tgt_output = generate_batch(batch_size, seq_len)
    src = src.to(device)
    tgt_input = tgt_input.to(device)
    tgt_output = tgt_output.to(device)

    # 生成 look-ahead mask，形状 [1,1,seq_len,seq_len]
    tgt_mask = subsequent_mask(seq_len).to(device)

    # 前向传播
    logits = model(src, tgt_input, None, tgt_mask)  # [batch, seq_len, vocab_size]

    # 计算损失
    loss = criterion(
        logits.reshape(-1, vocab_size),
        tgt_output.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step:4d}, loss = {loss.item():.4f}")

print("训练完成！\n")

# 测试模型复制长句子的能力
model.eval()

src_test = src_sentence.unsqueeze(0).to(device)  # [1, seq_len]
tgt_input_test = tgt_input_sentence.unsqueeze(0).to(device)  # [1, seq_len]
tgt_mask_test = subsequent_mask(seq_len).to(device)

with torch.no_grad():
    logits = model(src_test, tgt_input_test, None, tgt_mask_test)
    preds = logits.argmax(dim=-1)  # [1, seq_len]

print("输入句子: ", sentence)
print("模型输出: ", tensor_to_text(preds[0]))



if __name__ == "__main__":
    hf_run_all_verifications()
