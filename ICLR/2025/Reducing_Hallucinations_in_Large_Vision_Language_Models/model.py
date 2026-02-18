"""
Reducing Hallucinations in VLMs
减少视觉语言模型幻觉
"""

import torch
import torch.nn as nn

class HallucinationReducer(nn.Module):
    def __init__(self, vlm_model):
        super().__init__()
        self.vlm = vlm_model
        self.uncertainty_estimator = nn.Linear(4096, 1)
        
    def forward(self, image, text):
        # VLM输出
        output = self.vlm(image, text)
        
        # 不确定性估计
        uncertainty = self.uncertainty_estimator(output)
        
        # 置信度校准
        calibrated = output * torch.sigmoid(uncertainty)
        
        return calibrated, uncertainty
