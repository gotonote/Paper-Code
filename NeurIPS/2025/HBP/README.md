# HBP: Hierarchical Balance Packing

> NeurIPS 2025

## 论文信息

- **标题**: Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM
- **作者**: ModelTC
- **链接**: [GitHub](https://github.com/ModelTC/HBP)

## 核心贡献

1. 提出层次化平衡打包方法
2. 高效处理长上下文LLM的微调
3. 大幅降低显存占用

## 方法概述

HBP通过层次化的token打包策略，在保持训练效果的同时显著提高效率。

## 运行方式

```bash
pip install -r requirements.txt
python finetune.py --model llama2-7b --context 32768
```

## 依赖

- PyTorch >= 2.0
- transformers
- flash-attn
