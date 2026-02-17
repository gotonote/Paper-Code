"""X-Dyna模型"""
import torch
import torch.nn as nn

class XDynamic(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.motion_encoder = nn.Linear(55, 256)
        self.decoder = nn.ConvTranspose2d(320, 3, 7, stride=2, padding=3, output_padding=1)
        
    def forward(self, image, motion):
        img_feat = torch.relu(self.image_encoder(image))
        motion_feat = torch.relu(self.motion_encoder(motion))
        B, C, H, W = img_feat.shape
        motion_feat = motion_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)
        fused = torch.cat([img_feat, motion_feat], dim=1)
        return torch.sigmoid(self.decoder(fused))
