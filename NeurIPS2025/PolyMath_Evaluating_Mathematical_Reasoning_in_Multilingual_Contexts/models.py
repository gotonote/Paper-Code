"""PolyMath模型"""
import torch
import torch.nn as nn

class PolyMathModel(nn.Module):
    def __init__(self, vocab_size=30000):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 8), 6)
        self.classifier = nn.Linear(256, vocab_size)
        
    def forward(self, x):
        enc = self.encoder(x)
        return self.classifier(enc.mean(dim=1))
