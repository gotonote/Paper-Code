"""BASRec训练"""
import torch

class BASRecTrainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        users, items, labels = batch
        scores = self.model(users, items)
        loss = torch.nn.functional.cross_entropy(scores, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
