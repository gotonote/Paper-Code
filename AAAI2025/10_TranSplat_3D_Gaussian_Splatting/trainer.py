"""TranSplat训练"""
import torch

class TranSplatTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        images, gaussians = batch
        pred = self.model(images)
        loss = torch.nn.functional.mse_loss(pred, gaussians)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
