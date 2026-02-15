"""
Visual Reasoning 实现

本实现为第三方复现，仅供参考
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class VisualReasoningModel(nn.Module):
    """视觉推理模型"""
    
    def __init__(self, llm_name="gpt2"):
        super().__init__()
        
        # 视觉编码
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 语言模型
        self.llm = AutoModel.from_pretrained(llm_name)
        self.llm_dim = self.llm.config.hidden_size
        
        # 投影层
        self.vision_projection = nn.Linear(256, self.llm_dim)
        
        # 推理头
        self.reasoning_head = nn.Linear(self.llm_dim, 1)
    
    def forward(self, images, input_ids, attention_mask):
        # 视觉特征
        vision_features = self.vision_encoder(images)
        vision_features = vision_features.flatten(1)
        vision_features = self.vision_projection(vision_features)
        
        # 语言特征
        language_outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        
        # 融合
        combined = vision_features.unsqueeze(1) + language_outputs.last_hidden_state
        
        # 推理
        reasoning_logit = self.reasoning_head(combined).squeeze(-1)
        
        return reasoning_logit


def create_reasoning_model():
    return VisualReasoningModel()
