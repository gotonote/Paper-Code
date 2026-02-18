"""
TokenFlow: Unified Image Tokenizer
统一图像分词器
"""

import torch
import torch.nn as nn

class TokenFlowTokenizer(nn.Module):
    def __init__(self, codebook_size=8192):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
        )
        
        self.codebook = nn.Embedding(codebook_size, 256)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
        )
        
    def forward(self, x):
        feat = self.encoder(x)
        B, C, H, W = feat.shape
        feat_flat = feat.flatten(2).transpose(1, 2)
        indices = torch.argmax(feat_flat @ self.codebook.weight.T, dim=-1)
        quantized = self.codebook(indices)
        
        x = quantized.transpose(1, 2).reshape(B, C, H, W)
        recon = self.decoder(x)
        return recon, indices
