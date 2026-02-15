# Video Depth Anything: 超长视频深度估计 (CVPR 2025 Highlight)

## 论文信息

- **标题**: Video Depth Anything: Consistent Depth Estimation for Super-Long Videos
- **作者**: Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, Bingyi Kang
- **机构**: ByteDance
- **会议**: CVPR 2025 Highlight
- **论文链接**: https://arxiv.org/abs/2501.12375
- **项目主页**: https://videodepthanything.github.io

## 核心贡献总结

1. **超长视频处理**: 可应用于任意长度的视频，不影响质量、一致性或泛化能力
2. **高效推理**: 比扩散模型更快，参数更少，一致性深度精度更高
3. **度量深度**: 支持视频度量深度估计
4. **流式模式**: 支持无训练的流式视频深度估计

## 方法概述

Video Depth Anything基于Depth Anything V2，核心创新在于：

1. **视频深度估计**: 不牺牲质量的情况下处理超长视频
2. **时间一致性**: 保持帧间深度一致性
3. **快速推理**: 比扩散模型更快的推理速度
4. **零样本泛化**: 强大的泛化能力

### 关键技术点

- **Depth Anything V2**: 基于V2版本改进
- **Temporal Consistency**: 时间一致性技术
- **Streaming Mode**: 流式推理模式
- **Metric Depth**: 度量深度估计

## 代码结构说明

```
Video_Depth_Anything/
├── video_depth_anything/     # 核心模型代码
├── loss/                    # 训练损失函数
├── utils/                   # 工具函数
├── benchmark/               # 基准测试
├── assets/                  # 资源文件
├── run.py                   # 推理脚本
├── run_streaming.py         # 流式推理
├── app.py                   # Gradio App
└── README_en.md            # 英文原版README
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt

# 下载权重
bash get_weights.sh
```

### 推理示例

```python
import torch
from video_depth_anything import VideoDepthAnything

# 加载模型
model = VideoDepthAnything("vda-s")
model.eval()

# 视频深度估计
video_frames = [...]  # 视频帧列表
with torch.no_grad():
    depth_map = model(video_frames)
```

## 模型规模

| 模型 | 类型 | 下载 |
|:-----|:-----|:-----|
| VDA-S | 相对深度 | - |
| VDA-B | 相对深度 | - |
| VDA-L | 相对深度 | - |
| VDA-S-Metric | 度量深度 | - |
| VDA-B-Metric | 度量深度 | - |
| VDA-L-Metric | 度量深度 | - |

## 性能表现

在KITTI、NYUv2、ScanNet等基准数据集上取得领先性能。

## 来源

**来源：官方开源代码** - https://github.com/DepthAnything/Video-Depth-Anything

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
