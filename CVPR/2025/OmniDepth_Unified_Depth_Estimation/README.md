# OmniDepth: Unified Depth Estimation from Any Modality - CVPR 2025

## 论文信息

- **标题**: OmniDepth: Unified Depth Estimation from Any Modality
- **作者**: Xiaoyang Wu, Kai Chen et al.
- **链接**: https://github.com/omnidepth/omnidepth
- **会议**: CVPR 2025

## 核心贡献总结

1. 提出统一的深度估计框架，支持图像、点云、视频等多种输入模态
2. 引入跨模态注意力机制，实现模态间的信息共享
3. 在多个数据集上达到 SOTA 性能

## 方法概述

1. **多模态编码器**: 分别编码不同模态输入
2. **跨模态注意力**: 跨模态信息融合
3. **统一解码器**: 输出统一深度图

## 代码结构说明

```
OmniDepth/
├── encoder.py      # 多模态编码器
├── attention.py    # 跨模态注意力
├── decoder.py      # 统一解码器
├── train.py        # 训练脚本
├── requirements.txt
└── README.md
```

## 关键代码讲解

### 跨模态注意力

```python
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
        _, M, _ = x2.shape
        
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
```

### 统一解码器

```python
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
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(-1, x.shape[-1], 32, 32)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        depth = self.depth_head(x)
        
        return depth
```
