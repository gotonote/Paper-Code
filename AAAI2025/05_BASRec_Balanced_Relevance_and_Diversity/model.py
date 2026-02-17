"""
BASRec 模型定义
"""

import torch
import torch.nn as nn

class BASRecModel(nn.Module):
    """平衡相关性和多样性的推荐模型"""
    
    def __init__(self, user_num, item_num, embed_dim=64, num_heads=4):
        super().__init__()
        self.user_embed = nn.Embedding(user_num, embed_dim)
        self.item_embed = nn.Embedding(item_num, embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.diversity_scorer = nn.Linear(embed_dim, 1)
        
    def forward(self, users, items):
        user_emb = self.user_embed(users)
        item_emb = self.item_embed(items)
        
        # 计算相关性得分
        relevance = (user_emb * item_emb).sum(dim=-1)
        
        # 计算多样性得分
        item_features = self.diversity_scorer(item_emb)
        
        return relevance + 0.1 * item_features
