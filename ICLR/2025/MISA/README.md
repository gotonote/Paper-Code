# MISA: Advancing Prompt-Based Methods for Replay-Independent General Continual Learning

> ICLR 2025

## 论文信息

- **标题**: Advancing Prompt-Based Methods for Replay-Independent General Continual Learning
- **作者**: kangzhiq
- **链接**: [GitHub](https://github.com/kangzhiq/MISA)

## 核心贡献

1. 提出基于提示的通用持续学习方法
2. 不需要回放旧数据即可实现持续学习
3. 在多个基准数据集上取得SOTA表现

## 方法概述

MISA通过学习任务特定和任务通用的提示，实现高效且不遗忘的持续学习。

## 运行方式

```bash
pip install -r requirements.txt
python train.py --dataset cifar100 --n-tasks 10
```

## 依赖

- PyTorch >= 1.12
- torchvision
- numpy
