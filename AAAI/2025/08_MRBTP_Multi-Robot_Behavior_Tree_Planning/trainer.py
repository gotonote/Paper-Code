"""MRBTP训练"""
import torch

class MRBTPTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        states, actions = batch
        pred = self.model(states)
        loss = torch.nn.functional.cross_entropy(pred, actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
