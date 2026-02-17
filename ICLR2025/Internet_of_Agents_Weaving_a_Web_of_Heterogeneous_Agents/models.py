"""IoA模型"""
import torch
import torch.nn as nn

class AgentNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(100, 256)
        self.policy = nn.Linear(256, 10)
        
    def forward(self, x):
        return self.policy(torch.relu(self.encoder(x)))
