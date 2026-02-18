# FastVLM: 高效视觉编码的视觉语言模型 (CVPR 2025)

## 论文信息

- **标题**: FastVLM: Efficient Vision Encoding for Vision Language Models
- **作者**: Apple Team
- **机构**: Apple
- **会议**: CVPR 2025
- **论文链接**: https://www.arxiv.org/abs/2412.13303
- **项目主页**: https://apple.ml/fastvlm

## 核心贡献总结

1. **提出FastViTHD**: 新型混合视觉编码器，输出更少的token，显着减少高分辨率图像的编码时间
2. **超快TTFT**: 最小的变体比LLaVA-OneVision-0.5B快85倍的Time-to-First-Token (TTFT)
3. **小模型高效**: 仅使用3.4倍小的视觉编码器，性能更优
4. **移动端支持**: 提供iOS演示应用，展示模型在移动设备上的性能

## 方法概述

FastVLM是一种高效的视觉语言模型，核心创新在于FastViTHD（Fast Vision Transformer Hybrid Decoder）：

1. **混合视觉编码器**: 结合CNN和ViT的优点
2. **token压缩**: 输出更少的视觉token，减少计算量
3. **高分辨率支持**: 支持高分辨率图像的快速编码
4. **多规模版本**: 提供0.5B、1.5B、7B三种规模

### 关键技术点

- **FastViTHD架构**: 新型混合视觉编码器
- **Token减少策略**: 大幅减少视觉token数量
- **多阶段训练**: Stage 2和Stage 3两阶段训练
- **移动端优化**: 支持Apple Silicon和iOS设备

## 代码结构说明

```
FastVLM/
├── predict.py              # 推理脚本
├── trainer.py              # 训练器
├── get_models.sh           # 模型下载脚本
├── requirements.txt
└── README.md
```

## 快速开始

### 安装

```bash
conda create -n fastvlm python=3.10
conda activate fastvlm
pip install -e .
```

### 下载模型

```bash
bash get_models.sh   # 下载模型到 checkpoints 目录
```

### 推理示例

```bash
python predict.py --model-path /path/to/checkpoint-dir \
                  --image-file /path/to/image.png \
                  --prompt "Describe the image."
```

## 模型规模

| 模型 | 阶段 | 下载链接 |
|:-----|:----:|:--------:|
| FastVLM-0.5B | Stage 2 | [fastvlm_0.5b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage2.zip) |
| FastVLM-0.5B | Stage 3 | [fastvlm_0.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip) |
| FastVLM-1.5B | Stage 2 | [fastvlm_1.5b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage2.zip) |
| FastVLM-1.5B | Stage 3 | [fastvlm_1.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip) |
| FastVLM-7B | Stage 2 | [fastvlm_7b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage2.zip) |
| FastVLM-7B | Stage 3 | [fastvlm_7b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3.zip) |

## 关键代码讲解

### 推理代码

```python
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from PIL import Image

# 加载模型
model_name = "FastVLM"
model_path = "/path/to/checkpoint-dir"
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, 
    model_name,
    "llava-fastvithd"
)

# 准备图像
image = Image.open("image.png").convert("RGB")
image_tensor = process_images([image], image_processor, model.config)

# 推理
prompt = "Describe the image."
input_ids = tokenizer_image_token(prompt, tokenizer, ...)
with torch.no_grad():
    output = model.generate(input_ids)
```

## Apple Silicon支持

在Apple Silicon上运行需要导出模型格式，详细说明见 [`model_export`](model_export/) 目录。

提供预导出的模型：
- [fastvlm_0.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3_llm.fp16.zip)
- [fastvlm_1.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3_llm.int8.zip)
- [fastvlm_7b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3_llm.int4.zip)

## 依赖

- Python 3.10+
- PyTorch
- transformers
- PIL

## 来源

**来源：官方开源代码** - https://github.com/apple/ml-fastvlm

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
