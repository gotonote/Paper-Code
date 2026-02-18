"""DiT训练代码"""
import torch

class DiTTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        x, t = batch
        noise = torch.randn_like(x)
        predicted = self.model(x, t)
        loss = torch.nn.functional.mse_loss(predicted, noise)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
