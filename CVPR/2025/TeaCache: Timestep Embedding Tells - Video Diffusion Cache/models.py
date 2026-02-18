"""TeaCache模型"""
import torch
import torch.nn as nn

class CacheModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = {}
        
    def forward(self, x, t):
        key = int(t.item() * 10)
        if key in self.cache:
            return self.cache[key]
        self.cache[key] = x
        return x
