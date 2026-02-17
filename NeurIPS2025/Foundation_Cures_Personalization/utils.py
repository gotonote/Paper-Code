"""
个性化适配模块
用于微调基础模型
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np


class PersonaAdapter(nn.Module):
    """人格适配器"""
    def __init__(self, embed_dim: int = 768, adapter_dim: int = 64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, embed_dim)
        )
        
    def forward(self, x):
        return x + self.adapter(x)


class UserProfileEncoder(nn.Module):
    """用户画像编码器"""
    def __init__(self, num_features: int = 100, embed_dim: int = 768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        
    def forward(self, profile_features):
        return self.encoder(profile_features)


class Personalizer:
    """个性化器"""
    def __init__(self, base_model, adapter_dim: int = 64):
        self.base_model = base_model
        self.adapters = nn.ModuleDict()
        
    def add_user(self, user_id: str, profile_features: torch.Tensor):
        """添加用户"""
        adapter = PersonaAdapter(embed_dim=768, adapter_dim=adapter_dim)
        self.adapters[user_id] = adapter
        
    def personalize(self, user_id: str, input_ids: torch.Tensor, 
                   attention_mask: torch.Tensor) -> torch.Tensor:
        """个性化推理"""
        adapter = self.adapters.get(user_id)
        
        if adapter is None:
            return self.base_model(input_ids, attention_mask)
        
        outputs = self.base_model(input_ids, attention_mask, output_hidden_states=True)
        hidden = outputs.last_hidden_state
        
        adapted = adapter(hidden)
        
        return adapted


class PreferenceLearner:
    """偏好学习器"""
    def __init__(self, embed_dim: int = 768):
        self.preference_model = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def learn(self, positive_examples: List[torch.Tensor], 
              negative_examples: List[torch.Tensor]):
        """从示例学习偏好"""
        all_examples = positive_examples + negative_examples
        labels = [1.0] * len(positive_examples) + [0.0] * len(negative_examples)
        
        # 简化的训练
        print(f"从 {len(positive_examples)} 个正例和 {len(negative_examples)} 个负例学习")
        
    def predict_preference(self, hidden_states: torch.Tensor) -> float:
        """预测偏好分数"""
        return self.preference_model(hidden_states).item()


def create_personalized_model(base_model, user_profiles: Dict):
    """创建个性化模型"""
    personalizer = Personalizer(base_model)
    
    for user_id, profile in user_profiles.items():
        personalizer.add_user(user_id, profile)
        
    return personalizer
