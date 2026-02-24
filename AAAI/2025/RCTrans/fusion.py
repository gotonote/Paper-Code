import torch
import torch.nn as nn

class RadarCameraFusion(nn.Module):
    """雷达相机融合"""
    
    def __init__(self):
        super().__init__()
        self.radar_encoder = nn.Linear(5, 128)
        self.image_encoder = nn.Conv2d(3, 128, 3, padding=1)
        self.fusion = nn.MultiheadAttention(256, 8, batch_first=True)
        
    def forward(self, radar_data, image):
        radar_feat = self.radar_encoder(radar_data)
        image_feat = self.image_encoder(image).flatten(2).transpose(1, 2)
        fused, _ = self.fusion(radar_feat, image_feat, image_feat)
        return fused
