"""A-Mem训练"""
import torch

class AMemTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        memory, labels = batch
        scores = self.model(memory)
        loss = torch.nn.functional.cross_entropy(scores.squeeze(-1), labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
