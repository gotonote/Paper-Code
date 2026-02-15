# VGGT: Visual Geometry Grounded Transformer (CVPR 2025 Best Paper)

## 论文信息

- **标题**: VGGT: Visual Geometry Grounded Transformer
- **作者**: Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny
- **机构**: Visual Geometry Group, University of Oxford; Meta AI
- **会议**: CVPR 2025 (Best Paper Award)
- **论文链接**: https://arxiv.org/abs/2503.11651
- **项目主页**: https://vgg-t.github.io/

## 核心贡献总结

1. **端到端几何推理**: 提出VGGT，一个前馈神经网络，直接从单张、多张或数百张图像中推断出所有关键3D属性
2. **快速推理**: 可以在秒级时间内完成相机内外参、深度图、点图和3D点轨迹的估计
3. **泛化能力强**: 不需要额外的支撑网络（如深度估计器、特征匹配器），纯视觉几何 grounding
4. **支持商业使用**: 最新版本VGGT-1B-Commercial支持商业应用

## 方法概述

VGGT (Visual Geometry Grounded Transformer) 是一种基于Transformer的纯视觉几何推理模型：

1. **输入**: 单张或多张图像
2. **输出**: 相机内外参、深度图、点图、3D点轨迹
3. **核心思想**: 通过大规模数据预训练，学习图像中的几何结构表征

### 关键技术点

- **Vision Transformer**: 使用ViT作为图像编码器
- **Geometry Grounding**: 通过预训练学习几何表征
- **Camera Parameter Regression**: 直接回归相机内外参
- **Depth & Point Map Prediction**: 预测深度图和点图

## 代码结构说明

```
VGGT/
├── vggt/                    # 核心模型代码
│   ├── models/              # 模型定义
│   ├── utils/               # 工具函数
│   └── ...
├── training/                # 训练代码
├── examples/                # 示例代码
├── demo_gradio.py           # Gradio Demo
├── demo_colmap.py           # COLMAP格式导出
├── demo_viser.py            # Viser可视化
├── requirements.txt         # 依赖
└── README_en.md            # 英文原版README
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用示例

```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# 初始化模型并加载预训练权重
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# 加载图像
image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # 预测相机参数、深度图、点图等
        predictions = model(images)
```

## 关键代码讲解

### 模型初始化

```python
from vggt.models.vggt import VGGT

# 从HuggingFace加载预训练模型
model = VGGT.from_pretrained("facebook/VGGT-1B")
model.eval()

# 或手动加载权重
model = VGGT()
model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
))
```

### 图像预处理

```python
from vggt.utils.load_fn import load_and_preprocess_images

# 加载并预处理图像
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
images = load_and_preprocess_images(image_paths)
# images: [N, 3, H, W]
```

### 预测输出

```python
with torch.no_grad():
    predictions = model(images)

# predictions 包含:
# - camera_intrinsics: 相机内参
# - camera_extrinsics: 相机外参
# - depth: 深度图
# - point_map: 点图
# - x三维坐标点轨迹
```

## 依赖

- torch
- torchvision
- numpy
- Pillow
- huggingface_hub

## 来源

**来源：官方开源代码** - https://github.com/facebookresearch/vggt

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
