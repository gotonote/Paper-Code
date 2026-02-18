"""
CDPruner训练代码
"""
import torch

class PrunerTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        tokens, condition = batch
        pruned, mask = self.model.prune(tokens, condition)
        loss = -mask.float().mean()  # 鼓励剪枝
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
