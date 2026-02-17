"""
CDPruner: Conditional Diversity Token Pruning
条件多样性Token剪枝
"""

import torch
import torch.nn as nn

class CDPruner(nn.Module):
    def __init__(self, num_tokens=256):
        super().__init__()
        self.num_tokens = num_tokens
        
        # 重要性评分
        self.importance_scorer = nn.Linear(768, 1)
        
        # 多样性评分
        self.diversity_scorer = nn.Linear(768, 128)
        
        # 条件门控
        self.condition_gate = nn.Sequential(
            nn.Linear(128 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def compute_importance(self, tokens):
        """计算Token重要性"""
        return self.importance_scorer(tokens)
    
    def compute_diversity(self, tokens):
        """计算Token多样性"""
        feat = self.diversity_scorer(tokens)
        # 成对余弦相似度
        norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        sim = norm @ norm.T
        diversity = 1 - sim.mean()
        return diversity
    
    def prune(self, tokens, condition, threshold=0.5):
        """剪枝"""
        importance = self.compute_importance(tokens)
        diversity = self.compute_diversity(tokens)
        
        # 条件门控
        cond_feat = torch.cat([condition.expand_as(tokens[:,:,:128]), tokens], dim=-1)
        gate = self.condition_gate(cond_feat)
        
        # 组合分数
        score = importance * gate.squeeze(-1)
        
        # 保留高分数Token
        mask = score > threshold
        pruned_tokens = tokens * mask.unsqueeze(-1)
        
        return pruned_tokens, mask
