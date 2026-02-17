"""
DMAD: Diffusion Models for Anomaly Detection
异常检测扩散模型
"""

import torch
import torch.nn as nn

class AnomalyDiffusion(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
        )
        
        # 潜在空间
        self.latent = nn.Linear(256*8*8, 128)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(128, 256*8*8),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, 4, stride=2, padding=1),
        )
        
        # 异常评分
        self.anomaly_score = nn.Linear(128, 1)
        
    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        latent = self.latent(feat)
        recon = self.decoder(latent).reshape_as(x)
        
        # 重建误差作为异常分数
        score = self.anomaly_score(latent)
        
        return recon, score
