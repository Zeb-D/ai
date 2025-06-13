import math


class CustomOneCycleLR:
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, anneal_strategy='cos'):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.step_count = 0
        self.learning_rates = []

        # 计算上升和下降步数
        self.lr_max_step = int(total_steps * pct_start)
        self.lr_min_step = total_steps - self.lr_max_step

    def step(self):
        self.step_count += 1

        # 计算当前学习率
        if self.step_count <= self.lr_max_step:
            # 上升阶段
            progress = self.step_count / self.lr_max_step
            if self.anneal_strategy == 'cos':
                lr = self.max_lr * (1 + math.cos(math.pi * (1 - progress))) / 2
            else:  # linear
                lr = self.max_lr * progress
        else:
            # 下降阶段
            progress = (self.step_count - self.lr_max_step) / self.lr_min_step
            if self.anneal_strategy == 'cos':
                lr = self.max_lr * (1 + math.cos(math.pi * progress)) / 2
            else:  # linear
                lr = self.max_lr * (1 - progress)

        # 应用学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.learning_rates.append(lr)
        return lr
