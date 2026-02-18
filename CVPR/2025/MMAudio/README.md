# MMAudio: 高质量视频到音频合成 (CVPR 2025)

## 论文信息

- **标题**: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis
- **作者**: Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya, Alexander Schwing, Yuki Mitsufuji
- **机构**: University of Illinois Urbana-Champaign, Sony AI, Sony Group Corporation
- **会议**: CVPR 2025
- **论文链接**: https://arxiv.org/abs/2412.15322
- **项目主页**: https://hkchengrex.github.io/MMAudio

## 核心贡献总结

1. **多模态联合训练**: 创新性地采用多模态联合训练策略，允许在广泛的音视频和音频文本数据集上进行训练
2. **同步模块**: 引入同步模块，将生成的音频与视频帧对齐
3. **高质量音频生成**: 给定视频和/或文本输入，生成同步的高质量音频

## 方法概述

MMAudio是一种视频到音频合成模型，核心创新在于多模态联合训练：

1. **多模态输入**: 支持视频+文本、视频仅输入、文本仅输入等多种模式
2. **音视频同步**: 通过同步模块确保生成的音频与视频帧同步
3. **大规模数据训练**: 利用音视频和音频文本数据集进行联合训练

### 关键技术点

- ** Multimodal Joint Training**: 多模态联合训练策略
- **Synchronization Module**: 音频视频同步模块
- **Diffusion Model**: 基于扩散模型的音频生成
- **Temporal Alignment**: 时间对齐技术

## 代码结构说明

```
MMAudio/
├── demo.py               # 演示脚本
├── train.py              # 训练脚本
├── gradio_demo.py        # Gradio Demo
├── eval_onsets.py        # 评估脚本
├── requirements.txt
└── README.md
```

## 快速开始

### 安装

```bash
# 使用 miniforge 环境
conda create -n mmaudio python=3.9
conda activate mmaudio
pip install -e .
```

### 推理示例

```python
import torch
from mmaudio import MMAudioInference

# 初始化模型
model = MMAudioInference("mmaudio-v1")
model.eval()

# 从视频生成音频
video_path = "input_video.mp4"
output_audio = model(video_path=video_path)

# 从视频和文本生成音频
output_audio = model(video_path=video_path, text_prompt="rain falling")
```

## 预训练模型

| 模型 | 下载链接 |
|:-----|:--------:|
| MMAudio v1 | [HuggingFace](https://huggingface.co/hkchengrex/MMAudio/tree/main) |

## 依赖

- Python 3.9+
- PyTorch 2.5.1+
- torchaudio
- transformers

## 来源

**来源：官方开源代码** - https://github.com/hkchengrex/MMAudio

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
