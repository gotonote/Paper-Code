"""
MindJourney: World Models for Spatial Reasoning
世界模型 + 测试时扩展
"""

import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, state_dim=256, action_dim=64):
        super().__init__()
        
        # 状态编码
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # 动作编码
        self.action_encoder = nn.Linear(action_dim, 256)
        
        # 转换模型
        self.transition = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim)
        )
        
        # 奖励模型
        self.reward_head = nn.Linear(state_dim, 1)
        
    def forward(self, state, action):
        s = self.state_encoder(state)
        a = self.action_encoder(action)
        next_state = self.transition(torch.cat([s, a], dim=-1))
        reward = self.reward_head(next_state)
        return next_state, reward

class ReasoningAgent(nn.Module):
    def __init__(self, world_model: WorldModel):
        super().__init__()
        self.world_model = world_model
        self.policy = nn.Linear(256, 64)
        
    def think(self, observation, num_steps=10):
        """测试时推理"""
        state = self.world_model.state_encoder(observation)
        trajectory = [state]
        
        for _ in range(num_steps):
            action = self.policy(state)
            next_state, reward = self.world_model(state, action)
            trajectory.append(next_state)
            state = next_state
            
        return trajectory
