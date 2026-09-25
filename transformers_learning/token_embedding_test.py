import torch
import torch.nn as nn

torch.manual_seed(1234)


# 定义一个简单的模型，用于输入的 token ids 转换为 token embeddings
class TokenEmbedding(nn.Module):

    # 初始化模型的属性
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # 前向传播
    def forward(self, ids):
        return self.embedding(ids)


if __name__ == "__main__":
    vocab_size = 5012
    embedding_dim = 768
    model = TokenEmbedding(vocab_size, embedding_dim)
    print(model)

    # 随机指定一些 token ids
    ids = torch.tensor([[1, 3, 5, 9, 11]], dtype=torch.long)

    # 输出词嵌入
    token_embeddings = model(ids)
    print(token_embeddings)
