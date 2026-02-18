"""SelfElicit模型"""
import torch
import torch.nn as nn

class SelfElicitor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(768, 8), 6)
        self.generator = nn.Linear(768, 30000)
        
    def forward(self, x):
        enc = self.encoder(x)
        return self.generator(enc.mean(dim=1))
