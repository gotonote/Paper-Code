"""Wasserstein模型"""
import torch
import torch.nn as nn

class EntanglementMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(100, 256)
        self.metric = nn.Linear(256, 1)
        
    def forward(self, x, y):
        enc_x = torch.relu(self.encoder(x))
        enc_y = torch.relu(self.encoder(y))
        # Wasserstein距离近似
        diff = (enc_x - enc_y).pow(2).sum(dim=-1)
        return self.metric(diff.unsqueeze(-1))
