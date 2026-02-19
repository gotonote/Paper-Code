# T-CORE: Temporal Correspondence for Self-supervised Video Representation Learning

> CVPR 2025

## 论文信息

- **标题**: T-CORE: When the Future Becomes the Past: Taming Temporal Correspondence for Self-supervised Video Representation Learning
- **作者**: yafeng19
- **链接**: [GitHub](https://github.com/yafeng19/T-CORE)
- **原始论文**: [ArXiv](https://arxiv.org/abs/xxxx.xxxxx)

## 核心贡献

1. 提出了时间对应学习的新方法
2. 利用未来帧信息改进视频表征学习
3. 在多个视频理解任务上取得SOTA表现

## 方法概述

T-CORE通过创新的时间对应建模，利用未来信息来增强视频自监督学习的表征能力。

## 代码结构

```
T-CORE/
├── models/          # 模型架构
├── datasets/        # 数据集处理
├── train.py         # 训练脚本
└── requirements.txt # 依赖
```

## 运行方式

```bash
pip install -r requirements.txt
python train.py --dataset kinetics400
```

## 依赖

- PyTorch >= 1.12
- torchvision
- einops
