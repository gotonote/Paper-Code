import torch
import torch.nn as nn

class MotionDisentangler(nn.Module):
    """运动解耦器"""
    
    def __init__(self):
        super().__init__()
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1)
        )
        self.flow_predictor = nn.Conv2d(128, 2, 3, padding=1)
        
    def forward(self, frame1, frame2):
        x = torch.cat([frame1, frame2], dim=1)
        features = self.motion_encoder(x)
        flow = self.flow_predictor(features)
        return flow
