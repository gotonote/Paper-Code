"""
LongVU模型补充
"""
import torch
import torch.nn as nn

class LongVUModel(nn.Module):
    def __init__(self, num_frames=32):
        super().__init__()
        self.video_encoder = nn.Conv3d(3, 64, kernel_size=5, stride=2, padding=2)
        self.text_encoder = nn.Linear(768, 256)
        self.fusion = nn.MultiheadAttention(256, 8)
        self.predictor = nn.Linear(256, 1)
        
    def forward(self, video, text):
        # 视频编码
        B, T, C, H, W = video.shape
        video = video.view(B, C, T, H, W)
        v_feat = self.video_encoder(video)
        
        # 文本编码
        t_feat = self.text_encoder(text)
        
        # 融合
        fused, _ = self.fusion(v_feat.mean(dim=[2,3,4]).unsqueeze(0), t_feat.unsqueeze(0), t_feat.unsqueeze(0))
        
        return self.predictor(fused.squeeze(0))
