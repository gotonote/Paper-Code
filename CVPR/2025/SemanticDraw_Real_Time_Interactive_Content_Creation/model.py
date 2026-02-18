"""
SemanticDraw: Real-Time Interactive Content Creation
实时交互式内容创建
"""

import torch
import torch.nn as nn

class SemanticDraw(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 语义编码器
        self.sem_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
        )
        
        # 布局预测
        self.layout_head = nn.Sequential(
            nn.Linear(256*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # bounding box
        )
        
        # 渲染器
        self.renderer = nn.Sequential(
            nn.Linear(256+4, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        
    def forward(self, image, layout_hint=None):
        feat = self.sem_encoder(image).flatten(1)
        layout = self.layout_head(feat)
        
        # 渲染
        if layout_hint is not None:
            feat = torch.cat([feat, layout_hint], dim=-1)
        
        rendered = self.renderer(feat).sigmoid()
        return rendered, layout
