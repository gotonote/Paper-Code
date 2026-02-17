"""Magma训练代码"""
import torch

class MagmaTrainer:
    def __init__(self, model, lr=1e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        images, texts = batch
        output = self.model(images, texts)
        loss = output.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
