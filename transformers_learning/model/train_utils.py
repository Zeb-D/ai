import torch
import torch.nn as nn
from torch.autograd import Variable

class MultiGPULossCompute:
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        self.generator = generator
        # 注意：不要提前复制 criterion，单设备时直接使用即可
        self.criterion = criterion
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        # 如果只有一个设备（如 MPS 或 CPU），使用单设备路径
        if len(self.devices) == 1:
            # 前向计算
            logits = self.generator(out)
            # 计算损失（与多 GPU 分支保持一致的处理方式）
            loss = self.criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                targets.contiguous().view(-1)
            ) / normalize

            if self.opt is not None:
                loss.backward()
                self.opt.step()
                self.opt.optimizer.zero_grad()

            return loss.item() * normalize

        # 多 GPU 分支（保持原逻辑不变）
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]

            gen = nn.parallel.parallel_apply(generator, out_column)

            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            if self.opt is not None:
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return total * normalize

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        """
            初始化优化器包装类
            :param model_size: 模型的大小，通常是d_model的大小，用于计算学习率
            :param factor: 用于计算学习率的因子（通常是学习率的初始值）
            :param warmup: 预热步数，决定学习率从较小值到较大值的增长速度
            :param optimizer: 实际使用的优化器（例如Adam、SGD等）
        """
        self.optimizer = optimizer  # 存储实际的优化器（如Adam）
        self._step = 0  # 当前训练的步数
        self.warmup = warmup  # 预热步数
        self.factor = factor  # 学习率的因子
        self.model_size = model_size  # 模型的大小（通常是d_model，决定学习率的尺度）
        self._rate = 0  # 当前的学习率

    def step(self):
        """更新优化器的参数和学习率"""
        self._step += 1  # 增加当前步数
        rate = self.rate()  # 计算当前的学习率
        # 更新优化器中所有参数的学习率
        for p in self.optimizer.param_groups:
            p['lr'] = rate  # 设置当前学习率
        self._rate = rate  # 更新学习率
        self.optimizer.step()  # 执行一次优化步骤（更新参数）

    def rate(self, step=None):
        """根据当前步数计算学习率"""
        # 如果没有传入step，使用当前步数
        if step is None:
            step = self._step
        # 学习率计算公式：factor * (model_size ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    # 创建并返回一个NoamOpt优化器，包含Adam优化器作为基础
    return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))