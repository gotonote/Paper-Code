"""
Difix3D: Improving 3D Reconstructions with Diffusion Models
3D重建改进模型
"""

import torch
import torch.nn as nn

class Difix3DModel(nn.Module):
    """单步扩散3D重建模型"""
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # 扩散模块
        self.diffusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=4
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 7, padding=3)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        B, C, H, W = features.shape
        features_flat = features.flatten(2).transpose(1, 2)
        features_flat = self.diffusion(features_flat)
        features = features_flat.transpose(1, 2).reshape(B, C, H, W)
        return self.decoder(features)
