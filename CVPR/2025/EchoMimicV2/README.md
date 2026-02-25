# EchoMimicV2 - CVPR 2025

## 论文信息

- **标题**: EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation
- **作者**: Ant Group
- **链接**: https://github.com/antgroup/echomimic_v2
- **会议**: CVPR 2025 Highlight

## 核心贡献总结

1. 提出半身体动画生成框架
2. 简化的控制方式实现自然动作合成
3. 支持多种音频驱动的人物动画

## 方法概述

1. **音频特征提取**: 使用梅尔频谱图提取音频特征
2. **姿态生成**: 基于扩散模型生成人体姿态序列
3. **图像渲染**: 将姿态 seq2img 渲染到目标图像

## 代码结构说明

```
EchoMimicV2/
 audio_encoder.py   # 音频编码器
 pose_generator.py  # 姿态生成器
 renderer.py        # 图像渲染器
 requirements.txt  # 依赖
 README.md
```

## 关键代码讲解

```python
class AudioEncoder(nn.Module):
    """音频编码器"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gru = nn.GRU(64, 128, batch_first=True)
        
    def forward(self, mel_spectrogram):
        x = torch.relu(self.conv1(mel_spectrogram))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        features, _ = self.gru(x)
        return features


class PoseGenerator(nn.Module):
    """姿态序列生成器"""
    
    def __init__(self, audio_dim=128, pose_dim=55):
        super().__init__()
        self.lstm = nn.LSTM(audio_dim, 256, batch_first=True, num_layers=2)
        self.fc = nn.Linear(256, pose_dim)
        
    def forward(self, audio_features, num_frames=30):
        # 逐帧生成姿态
        poses = []
        h = None
        for t in range(num_frames):
            out, h = self.lstm(audio_features[:, t:t+1], h)
            pose = self.fc(out)
            poses.append(pose)
        return torch.cat(poses, dim=1)
```
