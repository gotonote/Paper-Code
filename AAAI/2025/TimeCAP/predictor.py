import torch
import torch.nn as nn

class TimeSeriesPredictor(nn.Module):
    """时间序列预测器"""
    
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, batch_first=True, num_layers=2)
        self.predictor = nn.Linear(128, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.predictor(out)
