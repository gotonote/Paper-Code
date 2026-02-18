"""采样模型"""
import torch
import torch.nn as nn

class DiversitySampler(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.scorer = nn.Linear(embed_dim, 1)
        
    def forward(self, embeddings, temperature=1.0):
        scores = self.scorer(embeddings) / temperature
        return torch.softmax(scores, dim=0)
