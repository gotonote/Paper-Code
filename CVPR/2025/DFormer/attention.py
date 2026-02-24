import torch.nn as nn

class GeometryAttention(nn.Module):
    """几何自注意力"""
    
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
    def forward(self, x, depth):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # 简化的注意力
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        return torch.matmul(attn, v)
