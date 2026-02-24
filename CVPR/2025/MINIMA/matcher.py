import torch

class ModalityInvariantMatcher:
    """模态不变匹配器"""
    
    def __init__(self):
        pass
        
    def extract_features(self, image):
        return torch.randn(1, 256, 32, 32)
        
    def match(self, feat1, feat2):
        # 特征匹配
        return torch.randint(0, 100, (50, 2))
