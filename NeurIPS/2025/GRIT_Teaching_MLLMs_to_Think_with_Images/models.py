"""GRIT模型"""
import torch
import torch.nn as nn

class GRITModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.text_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(768, 8), 6)
        self.fusion = nn.Linear(64*16*16 + 768, 1)
        
    def forward(self, image, text):
        img_feat = torch.relu(self.image_encoder(image)).flatten(1)
        txt_feat = self.text_encoder(text).mean(dim=1)
        return self.fusion(torch.cat([img_feat, txt_feat], dim=-1))
