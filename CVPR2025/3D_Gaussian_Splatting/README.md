# 3D Gaussian Splatting - CVPR 2025

## 论文信息

- **标题**: 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- **作者**: Bernhard Kerbl, Georgios Kopanas et al.
- **链接**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **会议**: CVPR 2025 Best Paper

## 核心贡献总结

1. 首次提出使用 3D 高斯函数表示辐射场，实现实时渲染
2. 保持可微渲染特性，支持端到端训练
3. 渲染速度比 NeRF 快 10-100 倍
4. 在新视角合成任务上达到 SOTA 质量

## 方法概述

1. **3D 高斯表示**: 使用 3D 高斯函数代替点云/网格表示
2. **自适应密度控制**: 根据梯度动态调整高斯分布数量
3. **球谐函数**: 使用球谐函数表示视角相关的颜色
4. **Tile-based 渲染**: 高效的 GPU 渲染策略

## 代码结构说明

```
3D_Gaussian_Splatting/
├── gaussian_model.py   # 高斯模型（核心类定义）
├── trainer.py          # 训练器
├── requirements.txt    # 依赖
└── README.md
```

## 关键代码讲解

### 3D 高斯模型

```python
class GaussianModel(nn.Module):
    """
    3D Gaussian Splatting 模型
    
    使用 3D 高斯函数表示场景，支持可微渲染
    """
    
    def __init__(self, num Gaussians=100000):
        super().__init__()
        
        # 3D 位置 (均值)
        self._xyz = nn.Parameter(torch.randn(num_Gaussians, 3))
        
        # 旋转 (四元数)
        self._rotation = nn.Parameter(torch.randn(num_Gaussians, 4))
        
        # 缩放
        self._scaling = nn.Parameter(torch.ones(num_Gaussians, 3))
        
        # 透明度
        self._opacity = nn.Parameter(torch.ones(num_Gaussians, 1))
        
        # 球谐系数 (视角相关颜色)
        self._features = nn.Parameter(torch.randn(num_Gaussians, 16, 3))
        
        # 激活函数
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = F.normalize
    
    def get_covariance(self):
        """
        从缩放和旋转计算协方差矩阵
        
        核心：3D 高斯的协方差 = R * S * S^T * R^T
        """
        # 缩放矩阵
        L = torch.diag_embed(self.scaling_activation(self._scaling))
        
        # 旋转矩阵 (从四元数)
        R = quaternion_to_rotation_matrix(self._rotation)
        
        # 协方差
        cov = R @ L @ L.transpose(-2, -1) @ R.transpose(-2, -1)
        
        return cov
```

### 高斯渲染

```python
def rasterize_gaussians(
    xyz,           # 高斯位置
    features,      # 球谐特征
    opacity,       # 透明度
    scaling,       # 缩放
    rotation,      # 旋转
    camera_matrix, # 相机矩阵
    image_size,    # 输出图像尺寸
):
    """
    高斯渲染
    
    核心：将 3D 高斯投影到 2D 图像平面
    """
    batch_size = xyz.shape[0]
    num_gaussians = xyz.shape[1]
    
    # 1. 计算 3D 协方差
    cov = compute_covariance_3d(scaling, rotation)
    
    # 2. 投影到 2D
    projected = project_to_2d(xyz, camera_matrix)
    cov_2d = project_covariance_3d_to_2d(cov, camera_matrix)
    
    # 3. 计算高斯权重
    pixel_coords = create_pixel_grid(image_size)
    weights = compute_gaussian_weights(pixel_coords, projected, cov_2d)
    
    # 4. 累积颜色
    colors = spherical_harmonics(features, view_direction)
    rendered_color = accumulate_color(weights, colors, opacity)
    
    return rendered_color
```
