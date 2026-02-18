# CDPruner: Maximizing Conditional Diversity for Token Pruning in MLLMs

## 论文信息

- **标题**: Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs
- **作者**: Theia Lab
- **会议**: NeurIPS 2025
- **GitHub**: [Theia-4869/CDPruner](https://github.com/Theia-4869/CDPruner)

## 核心内容

本文提出了 **CDPruner (Conditional Diversity Pruner)**，一种通过最大化条件多样性来剪枝多模态大型语言模型中 token 的方法。

### 主要贡献

1. **条件多样性**: 引入条件多样性指标来指导 token 剪枝。

2. **高效推理**: 在保持模型性能的同时显著减少计算量。

3. **即插即用**: 适用于多种 MLLM 架构的通用剪枝框架。

### 技术细节

- **多样性度量**: 基于条件熵的多样性计算
- **剪枝策略**: 自适应阈值选择
- **性能平衡**: 精度和效率的权衡优化

## 代码结构

```
code/
├── pruner/         # 剪枝器
├── models/         # 模型适配
├── evaluation/     # 评估代码
└── README.md       # 原始 README
```

## 使用方法

请参考原始 GitHub 仓库中的说明。

## 注意事项

本代码为官方开源实现。
