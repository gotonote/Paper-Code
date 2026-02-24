import torch
import torch.nn as nn

class PolypSegmenter(nn.Module):
    """息肉分割器"""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Conv2d(128, 1, 1)
        
    def forward(self, image, box):
        feat = self.encoder(image)
        mask = torch.sigmoid(self.decoder(feat))
        return mask
