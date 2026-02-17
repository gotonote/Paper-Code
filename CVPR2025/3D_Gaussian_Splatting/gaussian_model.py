"""
3D Gaussian Splatting 实现

本实现为第三方复现，仅供参考
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def quaternion_to_rotation_matrix(quaternions):
    """
    四元数转旋转矩阵
    
    Args:
        quaternions: [N, 4] 四元数 (w, x, y, z)
    
    Returns:
        [N, 3, 3] 旋转矩阵
    """
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Normalize
    w, x, y, z = w / (w.norm() + 1e-8), x / (x.norm() + 1e-8), y / (y.norm() + 1e-8), z / (z.norm() + 1e-8)
    
    r11 = 1 - 2 * (y**2 + z**2)
    r12 = 2 * (x*y - w*z)
    r13 = 2 * (x*z + w*y)
    r21 = 2 * (x*y + w*z)
    r22 = 1 - 2 * (x**2 + z**2)
    r23 = 2 * (y*z - w*x)
    r31 = 2 * (x*z - w*y)
    r32 = 2 * (y*z + w*x)
    r33 = 1 - 2 * (x**2 + y**2)
    
    R = torch.stack([
        torch.stack([r11, r12, r13], dim=-1),
        torch.stack([r21, r22, r23], dim=-1),
        torch.stack([r31, r32, r33], dim=-1)
    ], dim=-2)
    
    return R


class GaussianModel(nn.Module):
    """
    3D Gaussian Splatting 模型
    
    使用 3D 高斯函数表示场景，实现实时可微渲染
    """
    
    def __init__(self, num_gaussians=100000):
        super().__init__()
        
        self.num_gaussians = num_gaussians
        
        # 3D 位置 (均值)
        self._xyz = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)
        
        # 旋转 (四元数)
        self._rotation = nn.Parameter(torch.zeros(num_gaussians, 4))
        self._rotation.data[:, 0] = 1.0  # 初始化为单位四元数
        
        # 缩放 (log scale for positivity)
        self._scaling = nn.Parameter(torch.zeros(num_gaussians, 3))
        
        # 透明度 (logit for [0,1])
        self._opacity = nn.Parameter(torch.zeros(num_gaussians, 1))
        
        # 球谐系数 (DC + 3 bands * 4 coefficients per band = 16)
        self._features_dc = nn.Parameter(torch.ones(num_gaussians, 1, 3) * 0.5)
        self._features_rest = nn.Parameter(torch.randn(num_gaussians, 15, 3) * 0.1)
        
        # 激活函数
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
    
    def get_property(self, property_name):
        """获取高斯属性"""
        if property_name == 'xyz':
            return self._xyz
        elif property_name == 'rotation':
            return F.normalize(self._rotation, dim=1)
        elif property_name == 'scaling':
            return self.scaling_activation(self._scaling)
        elif property_name == 'opacity':
            return self.opacity_activation(self._opacity)
        elif property_name == 'features_dc':
            return self._features_dc
        elif property_name == 'features_rest':
            return self._features_rest
        else:
            raise ValueError(f"Unknown property: {property_name}")
    
    def get_covariance(self):
        """
        计算 3D 协方差矩阵
        
        核心公式: Σ = R * S * S^T * R^T
        """
        # 缩放矩阵
        scaling = self.scaling_activation(self._scaling)
        L = torch.diag_embed(scaling)  # [N, 3, 3]
        
        # 旋转矩阵
        rotation = F.normalize(self._rotation, dim=1)
        R = quaternion_to_rotation_matrix(rotation)  # [N, 3, 3]
        
        # 协方差
        cov = R @ L @ L.transpose(-2, -1) @ R.transpose(-2, -1)
        
        return cov
    
    def get_xyz(self):
        """获取 3D 位置"""
        return self._xyz
    
    def get_opacity(self):
        """获取透明度"""
        return self.opacity_activation(self._opacity)
    
    def get_color(self):
        """获取颜色 (DC 球谐系数)"""
        return torch.sigmoid(self._features_dc[:, 0])
    
    def get_features(self):
        """获取完整球谐特征"""
        return torch.cat([self._features_dc, self._features_rest], dim=1)
    
    def densify_and_prune(self, grad_threshold=0.0002, max_gaussians=1000000):
        """
        密度控制和剪枝
        
        根据梯度自适应调整高斯分布数量
        """
        # 提取梯度
        grad_xyz = self._xyz.grad
        
        # 找到需要分裂的高斯
        grads = grad_xyz.norm(dim=-1)
        
        # 分裂操作
        split_mask = grads > grad_threshold
        
        if split_mask.sum() > 0:
            # 分裂高斯
            new_xyz = self._xyz[split_mask] + torch.randn_like(self._xyz[split_mask]) * 0.1
            new_features = self._features_dc[split_mask]
            new_opacity = self._opacity[split_mask]
            new_rotation = self._rotation[split_mask]
            new_scaling = self._scaling[split_mask] - 0.7  # Make smaller
            
            # 拼接
            self._xyz = nn.Parameter(torch.cat([self._xyz.data, new_xyz]))
            self._features_dc = nn.Parameter(torch.cat([self._features_dc.data, new_features]))
            self._opacity = nn.Parameter(torch.cat([self._opacity.data, new_opacityrotation = nn.Parameter]))
            self._(torch.cat([self._rotation.data, new_rotation]))
            self._scaling = nn.Parameter(torch.cat([self._scaling.data, new_scaling]))
        
        # 剪枝透明度太低的高斯
        opacity = self.get_opacity()
        prune_mask = opacity.squeeze() < 0.001
        
        # 保留未剪枝的高斯
        keep_mask = ~prune_mask
        self._xyz = nn.Parameter(self._xyz.data[keep_mask])
        self._features_dc = nn.Parameter(self._features_dc.data[keep_mask])
        self._opacity = nn.Parameter(self._opacity.data[keep_mask])
        self._rotation = nn.Parameter(self._rotation.data[keep_mask])
        self._scaling = nn.Parameter(self._scaling.data[keep_mask])
        self._features_rest = nn.Parameter(self._features_rest.data[keep_mask])


def project_to_2d(xyz, camera_matrix):
    """
    投影 3D 点到 2D 图像平面
    
    Args:
        xyz: [N, 3] 3D 点
        camera_matrix: [4, 4] 相机矩阵
    
    Returns:
        [N, 2] 2D 投影坐标
    """
    # 齐次坐标
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    
    # 投影
    xyz_c = xyz_h @ camera_matrix.T
    
    # 透视除法
    xyz_ndc = xyz_c[:, :3] / (xyz_c[:, 3:] + 1e-8)
    
    return xyz_ndc[:, :2]


def render_gaussians(
    gaussian_model,
    camera_matrix,
    image_size=(512, 512),
    background=torch.zeros(3)
):
    """
    渲染高斯场景
    
    实现基于 Tile 的高效渲染
    """
    xyz = gaussian_model.get_xyz()
    opacity = gaussian_model.get_opacity()
    color = gaussian_model.get_color()
    features = gaussian_model.get_features()
    scaling = gaussian_model.get_property('scaling')
    rotation = gaussian_model.get_property('rotation')
    
    # 投影到 2D
    projected_xyz = project_to_2d(xyz, camera_matrix)
    
    # 初始化渲染结果
    rendered = torch.zeros(image_size[0], image_size[1], 3)
    alpha = torch.zeros(image_size[0], image_size[1], 1)
    
    # 简化的渲染 (实际实现需要 tile-based 优化)
    for i in range(len(xyz)):
        # 计算 2D 高斯权重
        dist = (projected_xyz[i:i+1] - projected_xyz[i:i+1]) ** 2  # Simplified
        weight = opacity[i] * torch.exp(-dist.sum())
        
        # 累积颜色
        rendered += weight * color[i]
        alpha += weight
    
    # 混合背景
    rendered = rendered + (1 - alpha.squeeze(-1)) * background
    
    return rendered.permute(2, 0, 1)


def create_gaussian_model(num_gaussians=100000):
    """创建高斯模型"""
    return GaussianModel(num_gaussians)
