# Video Generation with Diffusion Models - CVPR 2025

## 论文信息

- **标题**: Scalable and Efficient Video Generation
- **作者**: Sora Team, OpenAI
- **链接**: https://openai.com/sora
- **会议**: CVPR 2025

## 核心贡献总结

1. 提出基于扩散模型的视频生成框架，支持长视频生成
2. 引入时空联合注意力机制，有效建模视频的时序和空间信息
3. 提出高效的训练策略，支持高分辨率长视频生成
4. 在视频质量和运动一致性方面达到 SOTA

## 方法概述

1. **时空扩散架构**: 同时处理空间和时间维度
2. **3D VAE**: 将视频编码到潜空间进行高效处理
3. **时序注意力**: 使用因果注意力机制处理时序信息
4. ** Classifier-free Guidance**: 提高生成质量

## 代码结构说明

```
Video_Gen/
├── model/
│   ├── video_vae.py      # 3D VAE 模型
│   ├── diffusion.py      # 扩散模型
│   └── attention.py       # 时空注意力
├── train.py
├── sample.py
└── README.md
```

## 关键代码讲解

### 3D VAE 实现

```python
class VideoVAE(nn.Module):
    """
    3D Video VAE
    
    将视频编码到潜空间，支持高效的视频扩散训练
    """
    
    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            ResBlock3D(64),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            ResBlock3D(128),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            ResBlock3D(256),
            nn.Conv3d(256, latent_channels * 2, 3, padding=1)  # mean, logvar
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, 256, 3, padding=1),
            ResBlock3D(256),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            ResBlock3D(128),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            ResBlock3D(64),
            nn.Conv3d(64, in_channels, 3, padding=1)
        )
    
    def encode(self, x):
        """编码视频到潜空间"""
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z
    
    def decode(self, z):
        """从潜空间解码视频"""
        return self.decoder(z)
```

### 时空注意力机制

```python
class SpatiotemporalAttention(nn.Module):
    """
    时空联合注意力
    
    核心：同时建模空间关系和时间关系
    """
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # 因果掩码，确保时间信息只从过去流向未来
        self.causal_mask = None
    
    def forward(self, x, T):
        """
        Args:
            x: [B * T, N, C] - B*T 帧，每帧 N 个 patches
            T: 时间步数
        """
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        
        # 分离时间和空间维度
        x = x.view(B // T, T, H, W, C)
        
        # 空间注意力
        x_spatial = x.reshape(B // T, T * H * W, C)
        x_spatial = self.forward_attention(x_spatial)
        
        # 时间注意力 (因果)
        x_temp = x.permute(0, 2, 1, 3, 4).reshape(B // T, H * W, T, C)
        x_temp = self.forward_temporal_attention(x_temp)
        
        return x_spatial + x_temp
    
    def forward_attention(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
```

### 视频扩散训练

```python
def video_diffusion_loss(model, vae, x0, t):
    """
    视频扩散损失
    
    预测添加的噪声，支持可变长度视频
    """
    B, C, T, H, W = x0.shape
    
    # 编码到潜空间
    z0 = vae.encode(x0)
    
    # 采样噪声
    noise = torch.randn_like(z0)
    
    # 前向扩散
    alpha_bar = alphas_cumprod[t].view(B, 1, 1, 1, 1)
    zt = alpha_bar.sqrt() * z0 + (1 - alpha_bar).sqrt() * noise
    
    # 预测噪声
    predicted_noise = model(zt, t)
    
    return F.mse_loss(predicted_noise, noise)
```
