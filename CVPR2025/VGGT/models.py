"""VGGT模型"""
import torch
import torch.nn as nn

class VGGTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.depth_head = nn.Conv2d(64, 1, 3, padding=1)
        self.point_head = nn.Conv2d(64, 3, 3, padding=1)
        
    def forward(self, x):
        feat = torch.relu(self.encoder(x))
        depth = torch.sigmoid(self.depth_head(feat))
        points = self.point_head(feat)
        return depth, points
