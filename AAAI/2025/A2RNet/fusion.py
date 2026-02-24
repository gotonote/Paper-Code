import torch
import torch.nn as nn

class FusionNetwork(nn.Module):
    """图像融合网络"""
    
    def __init__(self):
        super().__init__()
        self.ir_encoder = nn.Conv2d(1, 64, 3, padding=1)
        self.rgb_encoder = nn.Conv2d(3, 64, 3, padding=1)
        self.fusion = nn.Conv2d(128, 64, 1)
        self.decoder = nn.Conv2d(64, 3, 3, padding=1)
        
    def forward(self, ir, rgb):
        ir_feat = torch.relu(self.ir_encoder(ir))
        rgb_feat = torch.relu(self.rgb_encoder(rgb))
        fused = torch.cat([ir_feat, rgb_feat], dim=1)
        fused = self.fusion(fused)
        return torch.sigmoid(self.decoder(fused))
