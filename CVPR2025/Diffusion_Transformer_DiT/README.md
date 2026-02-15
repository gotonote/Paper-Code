# Diffusion Transformer (DiT) - CVPR 2025

## 论文信息

- **标题**: Scalable Diffusion Models with Transformers
- **作者**: William Peebles, Saining Xie et al.
- **链接**: https://www.cs.cmu.edu/~diT/
- **会议**: CVPR 2025

## 核心贡献总结

1. 首次将 Transformer 架构引入扩散模型，替代传统的 UNet 结构
2. 提出 DiT 模型，证明 Transformer 可以很好地处理图像生成任务
3. 通过系统性实验分析发现模型规模是性能的关键因素
4. 在 ImageNet 256×256 达到 2.27 FID，超越所有现有扩散模型

## 方法概述

DiT 将扩散模型的骨干网络从 UNet 替换为 Vision Transformer (ViT)。核心设计：

1. **Patchify**: 将输入图像分割成固定大小的 patch 序列
2. **DiT Block**: 包含自适应层归一化 (AdaLN) 和交叉注意力机制
3. **时间步和类别条件注入**: 使用 AdaLN-Zero 实现高效条件注入
4. **渐进式训练策略**: 先在小分辨率训练，再提升到高分辨率

## 代码结构说明

```
DiT/
├── models/
│   ├── dit.py           # DiT 模型主体
│   ├── patch_embed.py   # Patch 嵌入层
│   └── blocks.py        # Transformer 块
├── train.py             # 训练脚本
├── sample.py            # 采样脚本
├── requirements.txt
└── README.md
```

## 运行方式

### 环境配置
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
python train.py --model dit-xl --batch_size 64 --epochs 100
```

### 生成图像
```bash
python sample.py --model dit-xl --num_images 50
```

## 关键代码讲解

### DiT 模型核心实现

```python
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) 模型
    
    核心思想：用 Transformer 替代 UNet 作为扩散模型的骨干网络
    """
    
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        
        # Patch 嵌入层：将图像转换为 patch 序列
        self.x_embedder = PatchEmbed(
            patch_size=patch_size, 
            in_chans=in_channels, 
            embed_dim=hidden_size
        )
        
        # 时间步嵌入 (Sinusoidal Position Embedding)
        self.t_embedder = TimestepEmbedding(hidden_size)
        
        # 类别嵌入
        self.y_embedder = nn.Embedding(num_classes, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # 输出层归一化
        self.final_layer = FinalLayer(hidden_size, self.out_channels, patch_size)
        
        self.initialize_weights()
    
    def unpatchify(self, x, h, w):
        """将 patch 序列还原为图像"""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h, w = h // p, w // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(x.shape[0], c, h * p, w * p)
    
    def forward(self, x, t, y):
        """
        前向传播
        
        Args:
            x: 带噪声的输入图像 [B, C, H, W]
            t: 时间步 [B]
            y: 类别标签 [B]
        
        Returns:
            预测的噪声 (和方差 if learn_sigma=True)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        
        # Patch 嵌入
        x = self.x_embedder(x)  # [B, num_patches, hidden_size]
        x = x + self.pos_embed  # 添加位置编码
        
        # 时间步和类别条件嵌入
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        cond = t_emb + y_emb
        
        # 通过 Transformer blocks
        for block in self.blocks:
            x = block(x, cond)
        
        # 最终输出层
        x = self.final_layer(x, t_emb)
        
        # 还原为图像
        x = self.unpatchify(x, h, w)
        
        return x
```

### DiT Block 实现 (关键创新点)

```python
class DiTBlock(nn.Module):
    """
    DiT Transformer Block
    
    核心创新：AdaLN-Zero 条件注入机制
    """
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, mlp_ratio)
        
        # AdaLN-Zero: 自适应层归一化 + 可学习的缩放因子
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
    
    def forward(self, x, cond):
        # 生成调制参数
        modulation = self.adaLN_modulation(cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=2)
        
        # 自适应归一化 + 注意力
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm, x_norm, x_norm)[0]
        
        # 自适应归一化 + MLP
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x
```

### 训练损失函数

```python
def train_step(model, x_start, t, noise):
    """
    扩散模型训练步骤
    
    使用简单的均方误差损失预测噪声
    """
    # 前向扩散：添加噪声
    t_expanded = t.view(-1, 1, 1, 1).float()
    sqrt_alpha_prod = extract(sqrt_alpha_prod, t, x_start.shape)
    sqrt_one_minus_alpha_prod = extract(one_minus_sqrt_alpha_prod, t, x_start.shape)
    
    noisy_images = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
    
    # 预测噪声
    predicted_noise = model(noisy_images, t)
    
    # MSE 损失
    loss = F.mse_loss(predicted_noise, noise, reduction='mean')
    
    return loss
```

## 参考文献

- Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers. ICCV 2023.
- 本实现为第三方复现，仅供参考
