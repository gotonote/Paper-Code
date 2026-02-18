# DGCPL: 概念先修关系学习的双图蒸馏 (IJCAI 2025)

## 论文信息

- **标题**: DGCPL: Dual Graph Distillation for Concept Prerequisite Relation Learning
- **作者**: Miao Zhang, Jiawei Wang, Jinying Han, Kui Xiao, Zhifei Li, Yan Zhang, Hao Chen, Shihui Wang
- **机构**: -
- **会议**: IJCAI 2025 Main Track on Natural Language Processing
- **论文链接**: -

## 核心贡献总结

1. **双图结构**: 提出双图结构，同时捕捉知识视角和学习行为视角
2. **概念-资源超图**: 构建概念-资源超图捕捉高阶知识特征
3. **学习行为图**: 构建学习行为图捕捉学习者行为特征
4. **门控蒸馏机制**: 使用门控知识蒸馏机制融合特征

## 方法概述

DGCPL是一种新型的深度学习模型，用于预测概念先修关系：

1. **双图结构**: 通过概念-资源超图和学习行为图构建双图结构
2. **超图神经网络**: 使用超图神经网络处理高阶关系
3. **有向图卷积**: 使用有向图卷积网络处理学习行为
4. **门控融合**: 使用门控机制融合不同视角的特征

### 关键技术点

- **Concept Prerequisite Learning**: 概念先修关系学习
- **Graph Distillation**: 图蒸馏
- **Hypergraph Neural Network**: 超图神经网络
- **Knowledge Distillation**: 知识蒸馏

## 代码结构说明

```
DGCPL/
├── data/                    # 数据目录
│   └── MOOC/              # MOOC数据集
├── load_data/              # 数据加载
├── utils/                  # 工具函数
│   ├── HyperGNN.py        # 超图神经网络
│   ├── DirectedGCN.py    # 有向图卷积网络
│   ├── SiameseNet.py     # 孪生网络
│   ├── model.py          # 模型定义
│   ├── train.py          # 训练脚本
│   └── test.py           # 测试脚本
└── README.md              # 英文原版README
```

## 快速开始

### 环境安装

```bash
# 创建conda环境
conda create -n dgcpl python=3.10
conda activate dgcpl

# 安装依赖
pip install torch==2.1.2+cu121
pip install torch-geometric==2.5.3
pip install scikit-learn==1.2.0
pip install pandas==1.5.3
```

### 训练

```bash
# 使用脚本训练
bash train.sh

# 或直接运行
python train.py \
 --in_channels 768 \
 --out_channels 256
```

### 测试

```bash
bash test.sh
```

## 来源

**来源：官方开源代码** - https://github.com/wisejw/DGCPL

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
