# DepthCrafter: Long Depth Sequence Generation

## 论文信息

- **论文标题**: DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos
- **作者**: Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, Ying Shan
- **机构**: Tencent AI Lab, HKUST, ARC Lab
- **论文链接**: https://arxiv.org/abs/2409.02095
- **项目主页**: https://depthcrafter.github.io
- **HuggingFace**: https://huggingface.co/spaces/tencent/DepthCrafter
- **代码仓库**: https://github.com/Tencent/DepthCrafter
- **奖项**: CVPR 2025 Highlight

## 核心贡献总结

1. **长深度序列生成**: 能够为开放世界视频生成时间一致的长深度序列，具有精细的细节。

2. **无额外信息需求**: 不需要额外的相机姿态或光流等额外信息。

3. **时序一致性**: 生成的深度序列具有优异的时间一致性。

4. **广泛适用**: 适用于各种开放世界视频。

## 方法概述

DepthCrafter的核心技术：

- **视频深度估计**: 基于深度学习的视频深度估计
- **时序建模**: 保持帧间深度的一致性
- **高效推理**: 优化推理速度，实现实时或近实时处理

## 代码结构说明

```
DepthCrafter_Long_Depth_Sequences/
├── depthcraft/                 # 核心模型代码
├── scripts/                   # 推理脚本
├── requirements.txt           # 依赖
└── README.md
```

## 运行方式

### 环境安装
```bash
pip install -r requirements.txt
```

### 推理
```bash
# 参考项目README进行推理
python inference.py --video_path input.mp4 --output_dir output/
```

## 依赖

- PyTorch
- opencv-python
- pillow
- numpy

## 许可证

研究用途
