import torch
import torch.nn as nn

class VideoCaptioner(nn.Module):
    """视频描述生成器"""
    
    def __init__(self):
        super().__init__()
        self.video_encoder = nn.LSTM(2048, 512, batch_first=True)
        self.caption_decoder = nn.LSTM(512, 512, batch_first=True)
        self.output = nn.Linear(512, 50000)  # 词汇表大小
        
    def forward(self, video_features):
        video_enc, _ = self.video_encoder(video_features)
        caption, _ = self.caption_decoder(video_enc)
        return self.output(caption)
