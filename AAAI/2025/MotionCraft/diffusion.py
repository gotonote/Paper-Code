import torch
import torch.nn as nn

class MotionDiffusion(nn.Module):
    """运动扩散模型"""
    
    def __init__(self):
        super().__init__()
        self.time_emb = nn.Linear(1, 128)
        self.motion_net = nn.GRU(128 + 512, 256, batch_first=True)
        self.output = nn.Linear(256, 55)  # 55个关节角
        
    def forward(self, noise, condition, timesteps):
        t_emb = self.time_emb(timesteps)
        out, _ = self.motion_net(torch.cat([condition, t_emb.unsqueeze(1).expand(-1, condition.shape[1], -1)], dim=-1))
        return self.output(out)
