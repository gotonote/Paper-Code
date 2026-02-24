import torch
import torch.nn as nn

class VideoHasher(nn.Module):
    """视频哈希器"""
    
    def __init__(self, hash_dim=64):
        super().__init__()
        self.encoder = nn.LSTM(2048, 512, batch_first=True)
        self.hash_layer = nn.Linear(512, hash_dim)
        
    def forward(self, video):
        encoded, _ = self.encoder(video)
        hash_bits = torch.tanh(self.hash_layer(encoded))
        return hash_bits
