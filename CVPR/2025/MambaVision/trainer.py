"""MambaVision训练代码"""
import torch

class MambaVisionTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train_step(self, batch):
        images = batch['image']
        labels = batch['label']
        output = self.model(images)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
