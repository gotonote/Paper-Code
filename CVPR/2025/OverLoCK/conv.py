import torch.nn as nn

class DynamicConv(nn.Module):
    """动态卷积"""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        return self.conv(x) * self.scale
