"""提示工程模型"""
import torch
import torch.nn as nn

class PromptOptimizer(nn.Module):
    def __init__(self, vocab_size=30000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        
    def forward(self, x):
        return self.embedding(x)
