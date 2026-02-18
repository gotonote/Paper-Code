"""ORMind模型"""
import torch
import torch.nn as nn

class ORSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(100, 256)
        self.decoder = nn.Linear(256, 50)
        
    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))
