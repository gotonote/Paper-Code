"""OneDiffusion模型"""
import torch
import torch.nn as nn

class UnifiedDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.decoder = nn.ConvTranspose2d(64, 3, 7, stride=2, padding=3, output_padding=1)
        
    def forward(self, x, t):
        feat = torch.relu(self.encoder(x))
        return self.decoder(feat)
