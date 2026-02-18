"""OrientAnything模型"""
import torch
import torch.nn as nn

class OrientationEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.orientation_head = nn.Linear(64*16*16, 3)  # Euler angles
        
    def forward(self, x):
        feat = torch.relu(self.encoder(x))
        return self.orientation_head(feat.flatten(1))
