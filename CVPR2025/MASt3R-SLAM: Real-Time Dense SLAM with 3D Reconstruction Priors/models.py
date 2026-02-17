"""MASt3R-SLAM模型"""
import torch
import torch.nn as nn

class SLAMSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.depth_head = nn.Conv2d(64, 1, 3, padding=1)
        self.pose_head = nn.Linear(64, 6)
        
    def forward(self, image):
        feat = torch.relu(self.encoder(image))
        depth = torch.sigmoid(self.depth_head(feat))
        pose = self.pose_head(feat.mean(dim=[2,3]))
        return depth, pose
