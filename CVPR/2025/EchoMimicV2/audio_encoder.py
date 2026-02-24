import torch
import torch.nn as nn

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
        x = x.permute(0, 2, 1)
        features, _ = self.gru(x)
        return features


class PoseGenerator(nn.Module):
    """姿态序列生成器"""
    
    def __init__(self, audio_dim=128, pose_dim=55):
        super().__init__()
        self.lstm = nn.LSTM(audio_dim, 256, batch_first=True, num_layers=2)
        self.fc = nn.Linear(256, pose_dim)
        
    def forward(self, audio_features, num_frames=30):
        poses = []
        h = None
        for t in range(num_frames):
            out, h = self.lstm(audio_features[:, t:t+1], h)
            pose = self.fc(out)
            poses.append(pose)
        return torch.cat(poses, dim=1)


class ImageRenderer(nn.Module):
    """图像渲染器"""
    
    def __init__(self):
        super().__init__()
        self.unet = nn.Sequential(
            nn.Conv2d(3 + 55, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, image, pose):
        # pose: (B, T, 55) -> (B, 55, H, W)
        b, t, d = pose.shape
        pose_map = pose[:, :, :22].transpose(1, 2).unsqueeze(3).expand(-1, -1, -1, image.shape[2]//8)
        pose_map = torch.nn.functional.interpolate(pose_map, size=(image.shape[2], image.shape[3]))
        x = torch.cat([image, pose_map], dim=1)
        return self.unet(x)
