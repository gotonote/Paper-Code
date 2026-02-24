import torch.nn as nn

class SparseCrossLayer(nn.Module):
    """稀疏跨层连接"""
    
    def __init__(self, dim):
        super().__init__()
        self.sparse_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        
    def forward(self, x1, x2):
        attn_out, _ = self.sparse_attn(x1, x2, x2)
        return attn_out + x1
