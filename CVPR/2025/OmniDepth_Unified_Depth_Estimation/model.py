import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    
    实现不同模态特征之间的信息交互
    """
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x1, x2):
        """
        x1: 模态1特征
        x2: 模态2特征
        """
        B, N, C = x1.shape
        M = x2.shape[1]
        
        # 计算QKV
        qkv = self.qkv(torch.cat([x1, x2], dim=1))
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 重塑为多头形式
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N + M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N + M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out


class UnifiedDecoder(nn.Module):
    """
    统一深度解码器
    
    从多模态特征预测深度图
    """
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.depth_head = nn.Conv2d(64, 1, 1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        depth = self.depth_head(x)
        
        return depth


class OmniDepth(nn.Module):
    """
    OmniDepth: 统一深度估计模型
    """
    
    def __init__(self, image_dim=768, point_dim=128, video_dim=512):
        super().__init__()
        
        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # 点云编码器
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # 视频编码器
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # 跨模态注意力
        self.cross_attn = CrossModalAttention(256)
        
        # 统一解码器
        self.decoder = UnifiedDecoder(256)
    
    def encode_image(self, x):
        return self.image_encoder(x)
    
    def encode_point(self, x):
        return self.point_encoder(x)
    
    def encode_video(self, x):
        return self.video_encoder(x)
    
    def forward(self, image_feat=None, point_feat=None, video_feat=None):
        # 获取各模态特征
        features = []
        if image_feat is not None:
            features.append(self.encode_image(image_feat))
        if point_feat is not None:
            features.append(self.encode_point(point_feat))
        if video_feat is not None:
            features.append(self.encode_video(video_feat))
        
        if len(features) == 0:
            return None
        
        # 融合特征
        fused = features[0]
        for i in range(1, len(features)):
            fused = self.cross_attn(fused, features[i])
        
        # 解码为深度图
        B, N, C = fused.shape
        H = W = int(N ** 0.5)
        fused = fused.transpose(1, 2).reshape(B, C, H, W)
        
        depth = self.decoder(fused)
        
        return depth


def compute_depth_loss(pred_depth, gt_depth):
    """计算深度估计损失"""
    # SILog 损失
    diff = torch.log(pred_depth) - torch.log(gt_depth)
    loss = torch.mean(diff ** 2)
    return loss


if __name__ == "__main__":
    model = OmniDepth()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    image_feat = torch.randn(2, 196, 768)
    output = model(image_feat=image_feat)
    print(f"Output shape: {output.shape}")
