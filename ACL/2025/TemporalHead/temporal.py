import torch
import torch.nn as nn

class TemporalHead(nn.Module):
    """时间头"""
    
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
    def forward(self, hidden_states, time_encoding):
        # 应用时间编码
        enhanced = hidden_states + time_encoding
        output, _ = self.temporal_attention(enhanced, enhanced, enhanced)
        return output
