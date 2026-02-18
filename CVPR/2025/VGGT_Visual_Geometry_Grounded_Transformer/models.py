"""VGGT-Visual Geometry模型"""
import torch
import torch.nn as nn

class VisualGeometryTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(64*16*16, 8), 6)
        self.point_head = nn.Linear(64*16*16, 3)
        
    def forward(self, x):
        feat = torch.relu(self.encoder(x)).flatten(1)
        trans = self.transformer(feat.unsqueeze(0))
        return self.point_head(trans.squeeze(0))
