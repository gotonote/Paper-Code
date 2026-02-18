"""Video Depth模型"""
import torch
import torch.nn as nn

class VideoDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.temporal = nn.LSTM(64, 64, 2, batch_first=True)
        self.depth_head = nn.Conv2d(64, 1, 3, padding=1)
        
    def forward(self, frames):
        B, T, C, H, W = frames.shape
        feat = torch.relu(self.encoder(frames.view(B*T, C, H, W)))
        feat = feat.view(B, T, -1).transpose(1, 2)
        temporal_out, _ = self.temporal(feat)
        depth = self.depth_head(temporal_out.transpose(1, 2).view(B, 64, H, W))
        return torch.sigmoid(depth)
