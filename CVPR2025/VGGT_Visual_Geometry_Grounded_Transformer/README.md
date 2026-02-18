# VGGT: Visual Geometry Grounded Transformer

## 论文信息

- **论文标题**: VGGT: Visual Geometry Grounded Transformer
- **作者**: Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny
- **机构**: Visual Geometry Group, University of Oxford; Meta AI
- **论文链接**: https://arxiv.org/abs/2503.11651
- **项目主页**: https://vgg-t.github.io/
- **代码仓库**: https://github.com/facebookresearch/vggt
- **奖项**: CVPR 2025 Best Paper Award

## 核心贡献总结

1. **前馈神经网络架构**: VGGT是一个前馈神经网络，能够直接从一张、几张或数百张图像中推断出场景的所有关键3D属性，包括内外相机参数、深度图、点云和3D轨迹。

2. **端到端3D重建**: 无需迭代优化或测试时调整，在秒级时间内完成3D重建。

3. **零样本单目重建**: 即使从未针对单目重建任务进行训练，模型也表现出惊人的单视图重建能力。

4. **高效推理**: 支持100张图像的3D重建仅需约3秒，200张图像约9秒。

## 方法概述

VGGT基于Transformer架构，利用视觉基础模型（DINOv2）的特征作为初始化，通过聚合多视图信息来预测相机的内外参数、深度图和点云。模型采用以下关键技术：

- **视觉特征提取**: 使用DINOv2作为骨干网络提取视觉特征
- **多视图聚合**: 通过Transformer聚合多视图信息
- **相机预测头**: 预测外参和内参矩阵（遵循OpenCV约定）
- **深度预测头**: 预测深度图及置信度
- **点云预测头**: 预测3D点坐标及置信度
- **轨迹预测头**: 预测图像间对应点的3D轨迹

## 代码结构说明

```
VGGT_Visual_Geometry_Grounded_Transformer/
├── models.py              # 核心模型代码
├── visual_util.py         # 可视化工具
├── demo_gradio.py         # Gradio交互式演示
├── demo_viser.py          # Viser 3D可视化
├── demo_colmap.py         # COLMAP格式导出
├── requirements.txt
└── README.md
```

## 运行方式

### 环境安装
```bash
cd VGGT_Visual_Geometry_Grounded_Transformer/code
pip install -r requirements.txt
```

### 快速推理
```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# 加载预训练模型
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# 加载图像
image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]
images = load_and_preprocess_images(image_names).to(device)

# 推理
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)
```

### 使用Gradio演示
```bash
python demo_gradio.py
```

### 导出为COLMAP格式
```bash
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --use_ba
```

## 关键代码讲解

### 1. 模型加载与预处理
```python
from vggt.utils.load_fn import load_and_preprocess_images

# 加载并预处理图像
images = load_and_preprocess_images(image_names).to(device)
# 返回: (B, N, C, H, W) - B为批次大小，N为图像数量
```

### 2. 相机参数解码
```python
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 从预测的pose encoding中提取内外参
extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
# extrinsic: (B, N, 4, 4) - 外部参数矩阵
# intrinsic: (B, N, 3, 3) - 内部参数矩阵
```

### 3. 深度图反投影
```python
from vggt.utils.geometry import unproject_depth_map_to_point_map

# 从深度图和相机参数重建3D点云
point_map = unproject_depth_map_to_point_map(
    depth_map.squeeze(0),  # (N, H, W)
    extrinsic.squeeze(0),   # (N, 4, 4)
    intrinsic.squeeze(0)   # (N, 3, 3)
)
```

### 4. 轨迹预测
```python
# 指定要跟踪的图像点
query_points = torch.FloatTensor([[100.0, 200.0], [60.72, 259.94]]).to(device)

# 预测3D轨迹
track_list, vis_score, conf_score = model.track_head(
    aggregated_tokens_list, images, ps_idx, 
    query_points=query_points[None]
)
```

## 依赖

- torch >= 2.0.0
- torchvision
- numpy
- Pillow
- huggingface_hub
- transformers
- opencv-python
- pycolmap (可选，用于COLMAP导出)
- viser (可选，用于3D可视化)
- gradio (可选，用于Web演示)

## 性能指标

在Co3D数据集上的表现（AUC@30°）:

| 模型 | AUC@30° |
|------|---------|
| VGGT-1B | 89.98 |

推理时间（单卡H100）:

| 输入帧数 | 时间(s) | 显存(GB) |
|----------|---------|----------|
| 1 | 0.04 | 1.88 |
| 10 | 0.14 | 3.63 |
| 100 | 3.12 | 21.15 |
| 200 | 8.75 | 40.63 |

## 许可证

代码采用MIT许可证。模型权重仅允许非商业用途，最新发布的VGGT-1B-Commercial版本允许商业使用。
