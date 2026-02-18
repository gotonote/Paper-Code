"""历史引导视频扩散"""
import torch
import torch.nn as nn

class HistoryGuidedDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.history_encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.decoder = nn.ConvTranspose2d(128, 3, 7, stride=2, padding=3, output_padding=1)
        
    def forward(self, x, history):
        feat = torch.relu(self.encoder(x))
        hist_feat = torch.relu(self.history_encoder(history))
        fused = torch.cat([feat, hist_feat], dim=1)
        return self.decoder(fused)
