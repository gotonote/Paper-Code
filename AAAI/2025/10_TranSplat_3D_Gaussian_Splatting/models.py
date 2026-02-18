"""TranSplat模型"""
import torch
import torch.nn as nn

class TranSplatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.gaussian_head = nn.Linear(64*16*16, 10000 * 7)
        
    def forward(self, x):
        feat = torch.relu(self.encoder(x)).flatten(1)
        return self.gaussian_head(feat)
