"""
世界模型推理框架
用于空间推理的测试时扩展
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class WorldState:
    """世界状态"""
    def __init__(self, position: np.ndarray, velocity: np.ndarray = None):
        self.position = position.astype(np.float32)
        self.velocity = velocity if velocity is not None else np.zeros(3)
        self.objects = []
        
    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([self.position, self.velocity]))
    
    def distance_to(self, other: 'WorldState') -> float:
        return np.linalg.norm(self.position - other.position)


class TransitionModel(nn.Module):
    """转移模型"""
    def __init__(self, state_dim: int = 6, action_dim: int = 3):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class RewardModel(nn.Module):
    """奖励模型"""
    def __init__(self, state_dim: int = 6, goal_dim: int = 3):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)


class ImaginaryActor:
    """想象Actor"""
    def __init__(self, transition: TransitionModel, reward: RewardModel):
        self.transition = transition
        self.reward = reward
        self.planning_horizon = 10
        
    def plan(self, start_state: WorldState, goal: np.ndarray,
              num_imaginations: int = 100) -> List[np.ndarray]:
        """想象规划"""
        best_path = None
        best_reward = -float('inf')
        
        goal_tensor = torch.from_numpy(goal).float()
        
        for _ in range(num_imaginations):
            path = [start_state.position.copy()]
            state = start_state.to_tensor().unsqueeze(0)
            
            total_reward = 0.0
            
            for _ in range(self.planning_horizon):
                # 随机动作
                action = torch.randn(1, 3) * 0.5
                
                # 想象一步
                next_state = self.transition(state, action)
                
                # 计算奖励
                r = self.reward(next_state, goal_tensor.unsqueeze(0))
                total_reward += r.item()
                
                state = next_state
                path.append(state.detach().numpy()[0, :3])
                
            if total_reward > best_reward:
                best_reward = total_reward
                best_path = path
                
        return best_path if best_path else [start_state.position]


class TestTimeScaler:
    """测试时扩展器"""
    def __init__(self, world_model: TransitionModel):
        self.world_model = world_model
        self.reasoning_buffer = deque(maxlen=1000)
        
    def think(self, observation: np.ndarray, goal: np.ndarray,
              compute_budget: int = 1000) -> np.ndarray:
        """思考"""
        # 扩展推理
        for _ in range(compute_budget):
            # 想象
            state = torch.from_numpy(observation).float().unsqueeze(0)
            action = torch.randn(1, 3) * 0.1
            next_state = self.world_model(state, action)
            
            self.reasoning_buffer.append({
                'state': state,
                'action': action,
                'next_state': next_state
            })
            
        # 聚合
        avg_action = torch.zeros(1, 3)
        for item in self.reasoning_buffer:
            avg_action += item['action']
        avg_action /= len(self.reasoning_buffer)
        
        return avg_action.detach().numpy()[0]


def create_reasoner():
    """创建推理器"""
    transition = TransitionModel()
    reward = RewardModel()
    actor = ImaginaryActor(transition, reward)
    scaler = TestTimeScaler(transition)
    return actor, scaler
