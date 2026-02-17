"""
Foundation Cures: Personalization
基础模型个性化
"""

import torch
import torch.nn as nn

class CuresModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.personalization_adapter = nn.Linear(768, 768)
        
    def forward(self, x, user_embed=None):
        base_out = self.base_model(x)
        
        if user_embed is not None:
            # 个性化调整
            adapted = self.personalization_adapter(user_embed)
            base_out = base_out + adapted
            
        return base_out
