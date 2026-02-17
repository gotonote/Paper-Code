"""
MAC模型定义
Multi-hop Attention for Context-aware Learning
"""

import torch
import torch.nn as nn

class MACModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),
            num_layers=num_layers
        )
        self.classifier = nn.Linear(embed_dim, 2)
        
    def forward(self, x):
        emb = self.embedding(x)
        encoded = self.encoder(emb)
        return self.classifier(encoded.mean(dim=1))
