# TinySAM: Pushing the Envelope for Efficient Segment Anything Model

> AAAI 2025

## 论文信息

- **标题**: TinySAM: Pushing the Envelope for Efficient Segment Anything Model
- **作者**: xinghaochen
- **链接**: [GitHub](https://github.com/xinghaochen/TinySAM)

## 核心贡献

1. 提出高效的端侧分割任意模型
2. 大幅降低计算量和参数量
3. 保持接近原始SAM的性能

## 方法概述

TinySAM通过知识蒸馏和结构优化，实现了在移动设备上的高效运行。

## 运行方式

```bash
pip install -r requirements.txt
python segment.py --image image.jpg
```

## 依赖

- PyTorch >= 2.0
- opencv-python
- numpy
