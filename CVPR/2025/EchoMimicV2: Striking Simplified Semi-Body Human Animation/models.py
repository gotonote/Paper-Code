"""EchoMimicV2模型"""
import torch
import torch.nn as nn

class HumanAnimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_encoder = nn.Linear(55, 256)
        self.image_encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.decoder = nn.ConvTranspose2d(320, 3, 7, stride=2, padding=3, output_padding=1)
        
    def forward(self, image, pose):
        img_feat = torch.relu(self.image_encoder(image))
        pose_feat = torch.relu(self.pose_encoder(pose))
        # 融合
        B, C, H, W = img_feat.shape
        pose_feat = pose_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)
        fused = torch.cat([img_feat, pose_feat], dim=1)
        return self.decoder(fused)
