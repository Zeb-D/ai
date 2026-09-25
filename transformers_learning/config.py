import torch

"""
模型参数配置
"""
# d_model = 512 表示模型的每个token的表示将使用512维的向量，这也决定了Transformer中间层的大小
d_model = 512
# 多头注意力机制中的头数。
n_heads = 8
# n_layers = 6表示模型中有6个Transformer编码器和解码器层。
n_layers = 6
# 自注意力机制中每个头的键(Key)向量的维度
d_k = 64
# 自注意力机制中每个头的值(Valve)向量的维度。
d_v = 64
# d_ff是前馈网络隐藏层的大小。d_ff = 2048表示前馈层的维度为2048。
d_ff = 2048
# dropout = 0.1表示在训练过程中，随机丢弃10%的神经元来避免模型过拟合。
dropout = 0.1

"""
词汇表和标记配置
"""
src_vocab_size = 32000
tgt_vocab_size = 32000
# padding_idx = 0表示填充token的索引为0，这通常用于填充短句，使得每个句子都具有相同的长度。
padding_idx = 0
# bos_idx = 2表示句子的开始符号(BOS)的索引是2。
bos_idx = 2
# eos_idx =3表示句子的结束符号(EOS)的索引是3
eos_idx = 3

"""
训练配置
"""
# 预计占用12GB
batch_size = 16
# 训练次数，在 5070 卡 大概1小时，在25-30 轮会收敛，Bleu Score: 26
# 在 mac m4 48G下，跑完一个批次需要 3个小时（主要卡住评估 BLEU分数）
epoch_num = 3
lr = 3e-4

"""
解码器和生成器配置
"""

# greed decode的最大句子长度# max_len = 60表示解码时生成的最大句子长度为60个token。
max_len = 60
# 在计算BLEU评分时使用的Beam Search的大小。
# beam_size =3表示在解码时，使用大小为3的Beam Search进行翻译
beam_size = 3

"""
文件路径和模型配置这些参数用于定义文件路径和是否加载预训练模型的设置
"""
data_dir = "./data"
train_data_path = "./dataset/train.json"
dev_data_path = "./dataset/dev.json"
test_data_path = "./dataset/test.json"

# 训练后，推理运行
translate_model_path = 'data/train/exp/weights/best_bleu_26.30.pth'
# 测试评估
test_model_path = 'data/train/exp33/weights/last_bleu_9.36.pth'
# 如果效果不好，接着跑
resume_path = 'data/train/exp1/weights/last_bleu_16.91.pth'
# 训练的模型产物目录
model_dir = "data/train"

gpu_id = ''
device_id = [0]
if gpu_id != "":
    device = torch.device("cuda:{}".format(gpu_id))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)
