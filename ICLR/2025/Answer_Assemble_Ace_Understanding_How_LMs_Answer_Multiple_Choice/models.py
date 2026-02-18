"""Answer Assemble模型"""
import torch
import torch.nn as nn

class AnswerAssembler(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(768, 8), 6)
        self.selector = nn.Linear(768, 4)
        
    def forward(self, options):
        encoded = self.encoder(options)
        return torch.softmax(self.selector(encoded.mean(dim=1)), dim=-1)
