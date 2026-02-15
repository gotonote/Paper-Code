# FoundationStereo: 零样本立体匹配 (CVPR 2025 Best Paper Nomination)

## 论文信息

- **标题**: FoundationStereo: Zero-Shot Stereo Matching
- **作者**: Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield
- **机构**: NVIDIA
- **会议**: CVPR 2025 Oral (Best Paper Nomination)
- **论文链接**: https://arxiv.org/abs/2501.09898
- **项目主页**: https://nvlabs.github.io/FoundationStereo/

## 核心贡献总结

1. **大规模合成数据集**: 构建了100万对立体图像对的大规模合成训练数据集，具有高度多样性和照片级真实感
2. **自筛选管道**: 设计自动自筛选管道去除模糊样本
3. **零样本泛化**: 实现强大的零样本泛化能力，在多个数据集上取得领先性能
4. **侧调优特征骨干**: 适应视觉基础模型的单目先验，缩小合成到真实的域差距

## 方法概述

FoundationStereo是一个用于立体深度估计的基础模型，核心创新在于：

1. **大规模合成数据**: 100万对立体图像对，多样性高且照片级真实
2. **自筛选管道**: 自动去除低质量样本
3. **侧调优骨干**: 将视觉基础模型的单目先验适应到立体匹配任务
4. **长程上下文推理**: 有效过滤代价卷

### 关键技术点

- **Side-Tuning**: 侧调优技术适应预训练特征
- **Cost Volume Filtering**: 代价卷过滤
- **Zero-Shot Generalization**: 零样本泛化能力
- **Synthetic Data**: 合成数据训练策略

## 代码结构说明

```
FoundationStereo/
├── core/                     # 核心模型代码
├── depth_anything/          # Depth Anything特征
├── dinov2/                  # DINOv2特征
├── scripts/                 # 训练推理脚本
├── docker/                  # Docker配置
├── Utils.py                 # 工具函数
├── environment.yml          # 环境配置
└── README_en.md            # 英文原版README
```

## 快速开始

### 安装

```bash
# 使用conda创建环境
conda env create -f environment.yml
conda activate foundation_stereo
```

### 推理示例

```python
import torch
from core import FoundationStereo

# 加载模型
model = FoundationStereo()
model.eval()

# 立体图像对推理
left_image = torch.randn(1, 3, 540, 960)
right_image = torch.randn(1, 3, 540, 960)

with torch.no_grad():
    disparity = model(left_image, right_image)
```

## 性能表现

- **Middlebury Leaderboard**: 全球第一
- **ETH3D Leaderboard**: 全球第一

## 商业模型

商业模型可在 [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationstereo) 获取。

## 来源

**来源：官方开源代码** - https://github.com/NVlabs/FoundationStereo

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
