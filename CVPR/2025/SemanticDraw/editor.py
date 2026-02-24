import torch
import torch.nn as nn

class SemanticEditor(nn.Module):
    """语义编辑器"""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 64, 3, padding=1)
        
    def forward(self, image, mask, edit_vector):
        """编辑图像"""
        x = torch.cat([image, mask], dim=1)
        return torch.sigmoid(self.conv(x))
