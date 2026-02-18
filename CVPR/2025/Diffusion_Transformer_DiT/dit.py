"""
DiT (Diffusion Transformer) 模型实现

本实现为第三方复现，仅供参考
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


class TimestepEmbedding(nn.Module):
    """时间步嵌入层"""
    
    def __init__(self, hidden_size, max_period=10000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.max_period = max_period
    
    def forward(self, t):
        t_emb = self.get_timestep_embedding(t, self.mlp[0].in_features)
        return self.mlp(t_emb)
    
    @staticmethod
    def get_timestep_embedding(timesteps, embedding_dim):
        """Sinusoidal position embedding"""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class Mlp(nn.Module):
    """MLP block"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiTBlock(nn.Module):
    """
    DiT Transformer Block
    
    核心创新：AdaLN-Zero 条件注入机制
    比传统的交叉注意力更高效
    """
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        
        # AdaLN-Zero: 自适应层归一化
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
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        # 自适应归一化 + MLP
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class FinalLayer(nn.Module):
    """最终输出层"""
    
    def __init__(self, hidden_size, out_channels, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )
    
    def forward(self, x, t_emb):
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=2)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    
    def __init__(self, patch_size=2, in_chans=4, embed_dim=1152):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) 模型
    
    用 Transformer 替代传统 UNet 作为扩散模型骨干
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
        
        # Patch 嵌入
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_size), requires_grad=False)
        
        # 条件嵌入
        self.t_embedder = TimestepEmbedding(hidden_size)
        self.y_embedder = nn.Embedding(num_classes, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # 输出层
        self.final_layer = FinalLayer(hidden_size, self.out_channels, patch_size)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化权重"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # 初始化 patch embedding
        nn.init.xavier_uniform_(self.x_embedder.proj.weight)
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        # 零初始化 adaLN 缩放因子
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
    
    def unpatchify(self, x, h, w):
        """将 patch 序列还原为图像"""
        p = self.patch_size
        c = self.out_channels
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
        
        # 添加位置编码
        pos_embed = self.pos_embed[:, :h*w, :]
        x = x + pos_embed
        
        # 条件嵌入
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        cond = t_emb + y_emb
        
        # 通过 Transformer blocks
        for block in self.blocks:
            x = block(x, cond)
        
        # 最终输出
        x = self.final_layer(x, t_emb)
        
        # 还原为图像
        x = self.unpatchify(x, h, w)
        
        return x
    
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Classifier-free guidance 前向传播
        """
        half = x.shape[0] // 2
        combined = torch.cat([x, x], dim=0)
        combined = combined + 0
        
        output = self.forward(combined, t, y)
        
        # 分离条件和非条件输出
        cond_output, uncond_output = output[:half], output[half:]
        
        # CFG: cond + scale * (cond - uncond)
        return uncond_output + cfg_scale * (cond_output - uncond_output)


def DiT_XL_2():
    """DiT-XL/2 模型配置"""
    return DiT(
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
    )


def DiT_L_2():
    """DiT-L/2 模型配置"""
    return DiT(
        patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
    )


def DiT_B_2():
    """DiT-B/2 模型配置"""
    return DiT(
        patch_size=2,
        hidden_size=768,
        depth=12,
        num_heads=12,
    )
