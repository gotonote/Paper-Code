# HarmoniCa: 扩散Transformer加速的特征缓存训练与推理协同 (ICML 2025)

## 论文信息

- **标题**: HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration
- **作者**: Yushi Huang, Zining Wang, Ruihao Gong, Jing Liu, Xinjie Zhang, Jinyang Guo, Xianglong Liu, Jun Zhang
- **机构**: ModelTC
- **会议**: ICML 2025
- **论文链接**: https://arxiv.org/pdf/2410.01723

## 核心贡献总结

1. **训练推理协同**: 提出训练-推理协同框架，实现 Diffusion Transformer 的块级缓存新SOTA
2. **逐步去噪训练(SDT)**: 确保去噪过程的连续性，利用前序步骤的信息
3. **图像误差代理目标(IEPO)**: 通过高效代理近似图像误差，平衡图像质量与缓存利用率
4. **显着加速**: 实现超过40%的延迟降低（理论加速2.07倍），训练时间减少25%

## 方法概述

HarmoniCa是一种新型的基于训练的扩散Transformer加速框架，核心创新在于：

1. **逐步去噪训练(SDT)**: 确保去噪过程的连续性，使前序步骤可以被有效利用
2. **图像误差代理目标(IEPO)**: 通过高效的代理来近似图像误差，在图像质量和缓存利用率之间取得平衡
3. **动态层缓存**: 在推理时实现动态特征缓存

### 关键技术点

- **Diffusion Transformer**: 扩散Transformer模型
- **Feature Caching**: 特征缓存
- **SDT (Step-Wise Denoising Training)**: 逐步去噪训练
- **IEPO (Image Error Proxy-Guided Objective)**: 图像误差代理目标

## 代码结构说明

```
HarmoniCa/
├── img/                    # 图像资源
├── train_router.py        # 训练路由模型
├── sample.py              # 推理采样
├── download.py            # 下载预训练模型
└── README.md              # 英文原版README
```

## 快速开始

### 环境安装

```bash
pip install accelerate diffusers timm torchvision wandb
python download.py
```

### 训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes=1 --nproc_per_node=4 --master_port 12345 train_router.py \
    --results-dir results \
    --model DiT-XL/2 \
    --image-size 256 \
    --num-classes 1000 \
    --epochs 2000 \
    --global-batch-size 64 \
    --wandb \
    --num-sampling-steps 10
```

### 推理

```bash
python sample.py \
    --model DiT-XL/2 \
    --vae ema \
    --image-size 256 \
    --cfg-scale 4 \
    --num-sampling-steps 10 \
    --accelerate-method dynamiclayer \
    --path Path/To/The/Trained/Router/ \
    --thres 0.1
```

## 性能表现

| 模型 | 加速比 | 图像质量 |
|:-----|:------:|:--------:|
| DiT-XL/2 256×256 | 1.44× | 几乎无损 |
| PixArt-Σ 2048×2048 | 1.73× | 几乎无损 |

## 来源

**来源：官方开源代码** - https://github.com/ModelTC/HarmoniCa

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
