# FlatQuant: Flatness Matters for LLM Quantization

> ICML 2025

## 论文信息

- **标题**: FlatQuant: Flatness Matters for LLM Quantization
- **作者**: ruikangliu
- **链接**: [GitHub](https://github.com/ruikangliu/FlatQuant)

## 核心贡献

1. 发现平坦性对LLM量化至关重要
2. 提出FlatQuant量化方法
3. 在极低比特下保持模型性能

## 方法概述

FlatQuant通过优化量化后的损失 landscape 平坦性，提高量化模型的泛化能力。

## 运行方式

```bash
pip install -r requirements.txt
python quantize.py --model llama2-7b --bits 4
```

## 依赖

- PyTorch >= 2.0
- transformers
- numpy
