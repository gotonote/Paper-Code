"""MRBTP模型"""
import torch
import torch.nn as nn

class BehaviorTreePlanner(nn.Module):
    def __init__(self, num_robots=4):
        super().__init__()
        self.state_encoder = nn.Linear(100, 256)
        self.planner = nn.LSTM(256, 256, 2, batch_first=True)
        self.action_head = nn.Linear(256, 10)
        
    def forward(self, states):
        enc = torch.relu(self.state_encoder(states))
        out, _ = self.planner(enc.unsqueeze(1))
        return self.action_head(out.squeeze(1))
