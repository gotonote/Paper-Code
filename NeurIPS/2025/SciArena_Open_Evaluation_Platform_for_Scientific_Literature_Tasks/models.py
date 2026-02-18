"""SciArena模型"""
import torch
import torch.nn as nn

class SciEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(768, 8), 6)
        self.classifier = nn.Linear(768, 100)
        
    def forward(self, x):
        encoded = self.encoder(x)
        return self.classifier(encoded.mean(dim=1))
