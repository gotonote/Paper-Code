"""AutoDiscovery模型"""
import torch
import torch.nn import nn

class SurpriseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(100, 256)
        self.surprise_head = nn.Linear(256, 1)
        
    def forward(self, x):
        return self.surprise_head(torch.relu(self.encoder(x)))
