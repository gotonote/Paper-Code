"""
Foundation Cures训练代码
"""
import torch

class CuresTrainer:
    def __init__(self, model, lr=1e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    def train(self, dataloader, epochs=3):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
        return self.model
