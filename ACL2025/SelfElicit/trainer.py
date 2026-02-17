"""SelfElicit训练"""
import torch

class SelfElicitTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        x, target = batch
        output = self.model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
