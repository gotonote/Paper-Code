"""UrBench模型补充"""
import torch
import torch.nn as nn

class UrBenchModel(nn.Module):
    def __init__(self, num_cameras=6):
        super().__init__()
        self.camera_encoders = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, 7, stride=2, padding=3), nn.ReLU())
            for _ in range(num_cameras)
        ])
        self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 8), 4)
        self.head = nn.Linear(256, 1000)
        
    def forward(self, images):
        feats = [enc(img) for enc, img in zip(self.camera_encoders, images)]
        fused = torch.stack(feats).mean(dim=0)
        return self.head(fused.mean(dim=[2,3]))
