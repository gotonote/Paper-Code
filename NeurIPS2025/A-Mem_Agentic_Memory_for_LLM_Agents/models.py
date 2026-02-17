"""A-Mem模型"""
import torch
import torch.nn as nn

class AgenticMemory(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.memory_encoder = nn.LSTM(embed_dim, embed_dim, 2, batch_first=True)
        self.memory_selector = nn.Linear(embed_dim, 1)
        
    def forward(self, memory):
        out, _ = self.memory_encoder(memory)
        scores = self.memory_selector(out)
        return scores
