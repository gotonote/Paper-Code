# MASt3R-SLAM - CVPR 2025

## 论文信息

- **标题**: MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors
- **作者**: R. Murai et al.
- **链接**: https://github.com/rmurai0610/MASt3R-SLAM
- **会议**: CVPR 2025

## 核心贡献总结

1. 首次将 MASt3R 3D 先验融入实时 SLAM 系统
2. 实现高质量稠密重建与实时性能的平衡
3. 具备深度估计和点云融合能力

## 方法概述

1. **3D先验融合**: 使用 MASt3R 提取场景几何先验
2. **实时跟踪**: 基于特征匹配的鲁棒相机跟踪
3. **稠密建图**: 融合多帧深度图生成完整场景

## 代码结构说明

```
MASt3R_SLAM/
├── mast3r_model.py   # MASt3R 模型
├── tracker.py        # 相机跟踪
├── mapper.py         # 稠密建图
├── requirements.txt  # 依赖
└── README.md
```

## 关键代码讲解

### MASt3R 特征提取

```python
class MASt3RExtractor:
    """
    MASt3R 特征提取器
    
    用于从图像中提取 3D 先验特征
    """
    
    def __init__(self, model_path):
        self.model = load_mast3r_model(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def extract_features(self, image):
        """
        提取图像特征和深度先验
        """
        # 图像预处理
        img_tensor = self.preprocess(image)
        
        # 前向传播
        with torch.no_grad():
            features = self.model(img_tensor.to(self.device))
            
        return {
            'descriptors': features['descriptors'],
            'depth': features['depth'],
            'conf': features['confidence']
        }
```
