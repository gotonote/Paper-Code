"""MedRAX模型"""
import torch
import torch.nn import nn

class MedicalReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.reasoner = nn.TransformerEncoder(nn.TransformerEncoderLayer(64*8*8, 8), 4)
        self.classifier = nn.Linear(64*8*8, 14)
        
    def forward(self, x):
        feat = torch.relu(self.encoder(x))
        feat = feat.flatten(1)
        reasoning = self.reasoner(feat.unsqueeze(0))
        return self.classifier(reasoning.squeeze(0))
