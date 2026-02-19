# EdgeTAM: On-Device Track Anything Model

> CVPR 2025

## 论文信息

- **标题**: EdgeTAM: On-Device Track Anything Model
- **作者**: Facebook Research
- **链接**: [GitHub](https://github.com/facebookresearch/EdgeTAM)
- **原始论文**: [ArXiv](https://arxiv.org/abs/xxxx.xxxxx)

## 核心贡献

1. 提出了端侧可运行的Track Anything Model
2. 优化了模型推理效率，适合移动设备部署
3. 保持了高质量的跟踪性能

## 方法概述

EdgeTAM通过模型压缩和高效架构设计，实现了在边缘设备上的实时目标跟踪。

## 代码结构

```
EdgeTAM/
├── models/          # 模型定义
├── utils/           # 工具函数
├── inference.py     # 推理脚本
└── requirements.txt # 依赖
```

## 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 运行推理
python inference.py --video input.mp4
```

## 依赖

- PyTorch >= 2.0
- OpenCV
- NumPy
