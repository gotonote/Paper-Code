import torch
import torch.nn as nn

class SteeringController(nn.Module):
    """ steering 控制器"""
    
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.steering_vector = nn.Parameter(torch.randn(hidden_dim))
        
    def apply_steering(self, hidden_states, strength=1.0):
        return hidden_states + strength * self.steering_vector
