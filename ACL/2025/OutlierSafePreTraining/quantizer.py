import torch
import torch.nn as nn

class OutlierSafeQuantizer(nn.Module):
    """异常值安全的量化器"""
    
    def __init__(self, bit_width=4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bit_width = bit_width
        
    def quantize(self, weight):
        scale = self.scale.abs()
        quantized = torch.round(weight / scale)
        quantized = torch.clamp(quantized, -(2**(self.bit_width-1)), 2**(self.bit_width-1)-1)
        return quantized * scale
