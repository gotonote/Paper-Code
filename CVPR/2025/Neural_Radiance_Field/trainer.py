"""NeRF训练代码"""
import torch

class NeRFTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        rays, pixels = batch
        rgb, depth = self.model(rays)
        loss = torch.nn.functional.mse_loss(rgb, pixels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
