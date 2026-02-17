"""
3D GS训练代码
"""
import torch

class GSTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        images, cameras = batch
        rendered = self.model(cameras)
        loss = torch.nn.functional.mse_loss(rendered, images)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
