"""StdGEN模型"""
import torch
import torch.nn as nn

class CharacterGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.sem_decoder = nn.ConvTranspose2d(64, 20, 7, stride=2, padding=3, output_padding=1)
        self.geo_decoder = nn.ConvTranspose2d(64, 3, 7, stride=2, padding=3, output_padding=1)
        
    def forward(self, x):
        feat = torch.relu(self.encoder(x))
        semantic = torch.softmax(self.sem_decoder(feat), dim=1)
        geometry = torch.sigmoid(self.geo_decoder(feat))
        return semantic, geometry
