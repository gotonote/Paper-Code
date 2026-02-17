"""视频生成训练代码"""
import torch

class VideoGenTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        frames = batch['video']
        noise = torch.randn_like(frames)
        predicted = self.model(frames, noise)
        loss = torch.nn.functional.mse_loss(predicted, noise)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
