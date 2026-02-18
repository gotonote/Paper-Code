"""GuardReasoner模型"""
import torch
import torch.nn import nn

class SafetyReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.reasoner = nn.TransformerEncoder(nn.TransformerEncoderLayer(64*16*16, 8), 4)
        self.safety_head = nn.Linear(64*16*16, 2)
        
    def forward(self, x):
        feat = torch.relu(self.encoder(x)).flatten(1)
        reasoning = self.reasoner(feat.unsqueeze(0))
        return self.safety_head(reasoning.squeeze(0))
