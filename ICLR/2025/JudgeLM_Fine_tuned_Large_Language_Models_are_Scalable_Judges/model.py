"""
JudgeLM: Scalable Judges
可扩展的LLM评判模型
"""

import torch
import torch.nn as nn

class JudgeLM(nn.Module):
    def __init__(self, base_model="llama-7b", embed_dim=4096):
        super().__init__()
        self.base_model = base_model
        # 评分头
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def judge(self, response_a, response_b, instruction):
        """评判两个响应"""
        # 简化的评判逻辑
        score_a = self.score_head(response_a)
        score_b = self.score_head(response_b)
        return score_a, score_b
    
    def forward(self, inputs):
        return self.judge(*inputs)
