"""
LADDER模型补充
"""
import torch
import torch.nn as nn

class LADDERModel(nn.Module):
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, 10)
        
    def forward(self, x):
        return self.classifier(self.encoder(x))
