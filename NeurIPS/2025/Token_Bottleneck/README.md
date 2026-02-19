# Token Bottleneck: One Token to Remember Dynamics

> NeurIPS 2025

## 论文信息

- **标题**: Token Bottleneck: One Token to Remember Dynamics
- **作者**: NAVER AI
- **链接**: [GitHub](https://github.com/naver-ai/tobo)

## 核心贡献

1. 提出Token瓶颈机制用于动态信息建模
2. 减少计算成本的同时保持模型性能
3. 适用于各种序列建模任务

## 方法概述

Token Bottleneck通过压缩token数量来提高效率，同时保留关键动态信息。

## 运行方式

```bash
pip install -r requirements.txt
python train.py --dataset cifar10
```

## 依赖

- PyTorch >= 1.12
- einops
