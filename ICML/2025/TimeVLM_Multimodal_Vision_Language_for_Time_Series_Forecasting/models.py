"""TimeVLM模型"""
import torch
import torch.nn as nn

class TimeVLModel(nn.Module):
    def __init__(self, num_series=10):
        super().__init__()
        self.series_encoder = nn.Linear(num_series, 256)
        self.text_encoder = nn.Linear(768, 256)
        self.predictor = nn.Linear(256, num_series)
        
    def forward(self, series, text):
        s_enc = torch.relu(self.series_encoder(series))
        t_enc = torch.relu(self.text_encoder(text))
        fused = s_enc * t_enc
        return self.predictor(fused)
