"""
NeRF (Neural Radiance Field) 实现

本实现为第三方复现，仅供参考
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, num_frequencies=10):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
    
    def forward(self, x):
        """编码坐标"""
        encoded = []
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)


class NeRF(nn.Module):
    """
    Neural Radiance Field
    
    使用 MLP 学习 5D 坐标到颜色和密度的映射
    """
    
    def __init__(
        self,
        input_dim=3,
        hidden_dim=256,
        num_layers=8,
        use_viewdirs=True,
        num_frequencies=10
    ):
        super().__init__()
        
        self.use_viewdirs = use_viewdirs
        self.input_dim = input_dim
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(num_frequencies)
        
        # 输入维度 = 坐标编码 + 视角编码 (if used)
        encoded_dim = self.pos_encoder.num_frequencies * 2 * input_dim
        if use_viewdirs:
            encoded_dim += self.pos_encoder.num_frequencies * 2 * 3
        
        # MLP 层
        layers = []
        for i in range(num_layers):
            in_dim = encoded_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        
        # 输出层
        self.density_layer = nn.Linear(hidden_dim, 1)
        self.color_layer = nn.Linear(hidden_dim, 3)
        
        # 视角相关颜色
        if use_viewdirs:
            self.feature_to_color = nn.Sequential(
                nn.Linear(hidden_dim + encoded_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)
            )
    
    def forward(self, xyz, viewdirs=None):
        """
        NeRF 前向传播
        
        Args:
            xyz: [N, 3] 3D 坐标
            viewdirs: [N, 3] 视角方向
        
        Returns:
            rgb: [N, 3] 颜色
            density: [N, 1] 密度
        """
        # 编码坐标
        encoded_xyz = self.pos_encoder(xyz)
        
        # MLP
        h = self.mlp(encoded_xyz)
        
        # 密度
        density = F.relu(self.density_layer(h))
        
        # 颜色
        if self.use_viewdirs and viewdirs is not None:
            encoded_viewdirs = self.pos_encoder(viewdirs)
            color_input = torch.cat([h, encoded_viewdirs], dim=-1)
            rgb = torch.sigmoid(self.feature_to_color(color_input))
        else:
            rgb = torch.sigmoid(self.color_layer(h))
        
        return rgb, density


def volume_render(rgb, density, depths, dirs):
    """
    体积渲染
    
    使用数值积分计算像素颜色
    """
    # 计算不透明度
    delta = depths[:, 1:] - depths[:, :-1]
    delta = torch.cat([delta, torch.ones_like(delta[:, :1])], dim=-1)
    
    # 不透明度累积
    opacity = 1.0 - torch.exp(-density * delta)
    
    # 颜色累积
    weights = opacity * torch.cumprod(
        torch.cat([torch.ones_like(opacity[:, :1]), 1.0 - opacity[:, :-1]], dim=-1),
        dim=-1
    )
    
    # 最终颜色
    rgb_final = (weights.unsqueeze(-1) * rgb).sum(dim=1)
    
    return rgb_final


def create_nerf_model():
    """创建 NeRF 模型"""
    return NeRF(input_dim=3, hidden_dim=256, num_layers=8, use_viewdirs=True)
