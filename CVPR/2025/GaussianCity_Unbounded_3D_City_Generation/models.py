"""GaussianCity模型"""
import torch
import torch.nn as nn

class CityGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(256, 10000 * 6)
        
    def forward(self, z):
        return self.decoder(z).view(-1, 10000, 6)
