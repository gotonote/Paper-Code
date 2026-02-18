"""MoA模型"""
import torch
import torch.nn as nn

class MixtureOfAgents(nn.Module):
    def __init__(self, num_agents=4):
        super().__init__()
        self.agents = nn.ModuleList([nn.Linear(768, 768) for _ in range(num_agents)])
        self.gate = nn.Linear(768, num_agents)
        
    def forward(self, x):
        outputs = [agent(x) for agent in self.agents]
        weights = torch.softmax(self.gate(x), dim=-1)
        return sum(w * o for w, o in zip(weights.T, outputs))
