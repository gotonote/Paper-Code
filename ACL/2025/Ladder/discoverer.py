import torch
import torch.nn as nn

class SliceDiscoverer(nn.Module):
    """切片发现器"""
    
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 10)
        
    def discover_slices(self, features, labels):
        # 识别具有特定模式的样本切片
        predictions = self.classifier(features)
        return predictions.argmax(dim=-1)
