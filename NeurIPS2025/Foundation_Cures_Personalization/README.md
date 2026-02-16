# Foundation Cures Personalization: Improving Personalized Models' Prompt Consistency

## 论文信息

- **标题**: Foundation Cures Personalization: Improving Personalized Models' Prompt Consistency via Hidden Foundation Models
- **作者**: Yiyang Cai 等
- **会议**: NeurIPS 2025
- **GitHub**: [YIYANGCAI/FreeCure](https://github.com/YIYANGCAI/FreeCure)

## 核心内容

本文提出了一种名为 **FreeCure** 的方法，通过隐藏基础模型来改进个性化模型的提示一致性。

### 主要贡献

1. **提示一致性**: 解决个性化模型在不同提示下表现不一致的问题。

2. **隐藏基础模型**: 引入隐藏基础模型的概念，保持个性化能力的同时提高泛化性。

3. **零样本迁移**: 方法支持零样本迁移到新任务，无需额外训练。

### 技术细节

- **模型架构**: 独特的双分支架构
- **训练策略**: 两阶段训练范式
- **一致性正则化**: 确保不同提示下的输出一致性

## 代码结构

```
code/
├── models/          # 模型架构
├── training/        # 训练代码
├── inference/      # 推理代码
└── README.md       # 原始 README
```

## 使用方法

请参考原始 GitHub 仓库中的说明。

## 注意事项

本代码为官方开源实现。
