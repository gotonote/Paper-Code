"""
MindJourney训练代码
"""
import torch

class MindJourneyTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def train(self, dataloader, epochs=3):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
        return self.model
