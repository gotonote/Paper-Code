"""
DMAD: Diffusion Models for Anomaly Detection
异常检测扩散模型 - 补充核心代码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class DiffusionUnet(nn.Module):
    def __init__(self, in_channels=3, time_dim=256):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 编码器
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # 中间层
        self.middle = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256)
        )
        
        # 解码器
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec3 = nn.Conv2d(64, in_channels, 3, padding=1)
        
    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_mlp(t.float())
        
        # 编码
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        
        # 中间
        m = self.middle(e3)
        
        # 解码
        d1 = F.relu(self.dec1(m))
        d2 = F.relu(self.dec2(d1))
        out = self.dec3(d2)
        
        return out

class AnomalyDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.anomaly_head = nn.Linear(256, 1)
        
    def forward(self, x, t=None):
        if t is None:
            t = torch.zeros(x.size(0), 1).to(x.device)
        
        features = self.backbone.enc3(F.relu(self.backbone.enc1(x)))
        features = features.mean(dim=[2, 3])
        
        score = self.anomaly_head(features)
        return score
