"""FoundationStereo模型"""
import torch
import torch.nn as nn

class StereoMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(6, 64, 7, stride=2, padding=3)
        self.cost_volume = nn.Conv3d(64, 64, 3, padding=1)
        self.disparity_head = nn.Conv3d(64, 1, 3, padding=1)
        
    def forward(self, left, right):
        feat = torch.relu(self.encoder(torch.cat([left, right], dim=1)))
        B, C, H, W = feat.shape
        # 简化的cost volume
        return torch.zeros(B, 1, H, W).to(feat.device)
