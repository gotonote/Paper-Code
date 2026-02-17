"""
全原子扩散训练代码
"""
import torch

class AtomDiffusionTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        atoms, coords = batch
        pred_coords, pred_vel = self.model(atoms, coords)
        loss = torch.nn.functional.mse_loss(pred_coords, coords)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
