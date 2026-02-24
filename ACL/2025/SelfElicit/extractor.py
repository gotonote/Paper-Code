import torch
import torch.nn as nn

class EvidenceExtractor(nn.Module):
    """证据提取器"""
    
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def extract(self, hidden_states):
        scores = self.attention(hidden_states)
        weights = torch.softmax(scores, dim=1)
        evidence = (hidden_states * weights).sum(dim=1)
        return evidence
