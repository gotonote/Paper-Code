"""FoundationStereo Zero-Shot"""
import torch
import torch.nn as nn

class StereoZeroShot(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(6, 64, 7, stride=2, padding=3)
        self.depth_head = nn.Conv2d(64, 1, 3, padding=1)
        
    def forward(self, left, right):
        feat = torch.relu(self.encoder(torch.cat([left, right], dim=1)))
        return torch.sigmoid(self.depth_head(feat))
