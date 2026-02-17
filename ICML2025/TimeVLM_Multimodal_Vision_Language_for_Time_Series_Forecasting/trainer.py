"""TimeVLM训练"""
import torch

class TimeVLMTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        series, text, target = batch
        output = self.model(series, text)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
