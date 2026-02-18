# 平衡LLM采样中的多样性与风险 (ACL 2025)

## 论文信息

- **标题**: Balancing Diversity and Risk in LLM Sampling
- **作者**: Zhou Yuxuan, etc.
- **机构**: -
- **会议**: ACL 2025 Main Paper
- **论文链接**: -

## 核心贡献总结

1. **评估基准**: 提出自适应采样的评估基准和用户指南
2. **风险-多样性权衡**: 系统性地分析不同截断采样方法在风险和多样性之间的权衡
3. **上下文保留前缀树**: 构建高效的上下文保留前缀树用于评估

## 方法概述

本文提出了一种针对自适应采样方法的系统评估框架：

1. **上下文保留前缀树(CP-Trie)**: 高效处理大规模数据集，构建上下文保留的前缀树
2. **双指标评估**: 使用Recall Mean（多样性）和Risk Standard Error（稳定性）两个指标
3. **公平比较**: 在相同的平均风险水平下比较不同方法的性能

### 关键技术点

- **Adaptive Sampling**: 自适应采样
- **CP-Tree**: 上下文保留前缀树
- **Diversity-Risk Tradeoff**: 多样性-风险权衡
- **Truncation Sampling**: 截断采样

## 代码结构说明

```
Balancing_Diversity_and_Risk_in_LLM_Sampling/
├── collect_prefix_tree.py     # 构建上下文保留前缀树
├── sort_and_visualize_prefix_tree.py  # 排序和可视化
├── estimate_optimal_truncation.py     # 估计最优截断
├── compute_truncation.py      # 计算截断
├── evaluate_recall_and_stability.py    # 评估召回率和稳定性
└── README.md                  # 英文原版README
```

## 快速开始

### 构建上下文保留前缀树

```bash
# 步骤1: 从数据集构建前缀树
python collect_prefix_tree.py

# 步骤2: 排序
python sort_and_visualize_prefix_tree.py
```

### 评估截断采样方法

```bash
# 步骤1: 估计最优截断
python estimate_optimal_truncation.py

# 步骤2: 计算不同方法的截断
python compute_truncation.py

# 步骤3: 评估召回率和稳定性
python evaluate_recall_and_stability.py
```

## 来源

**来源：官方开源代码** - https://github.com/ZhouYuxuanYX/Benchmarking-and-Guiding-Adaptive-Sampling-Decoding-for-LLMs

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
