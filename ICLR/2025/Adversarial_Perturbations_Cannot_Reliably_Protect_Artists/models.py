"""对抗扰动保护模型"""
import torch
import torch.nn as nn

class Protector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 3, padding=1)
        self.perturbation = nn.Conv2d(64, 3, 3, padding=1)
        
    def forward(self, x):
        feat = torch.relu(self.encoder(x))
        noise = self.perturbation(feat)
        return x + torch.tanh(noise)
