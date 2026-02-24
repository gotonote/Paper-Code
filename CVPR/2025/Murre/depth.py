import torch

class SfMGuidedDepth(nn.Module):
    """SfM引导的深度估计"""
    
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU()
        )
        
    def estimate_depth(self, image, poses, points_3d):
        """估计深度"""
        features = self.encoder(image)
        # 简化实现
        depth = torch.ones(features.shape[0], 1, features.shape[2], features.shape[3])
        return depth
