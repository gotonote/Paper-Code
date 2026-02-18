# GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning

## 论文信息

- **标题**: GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning
- **作者**: Yue Liu 等
- **会议**: NeurIPS 2025
- **GitHub**: [yueliu1999/GuardReasoner-VL](https://github.com/yueliu1999/GuardReasoner-VL)

## 核心内容

本文提出了 **GuardReasoner-VL**，一种通过强化推理来保护视觉语言模型 (VLMs) 的方法。

### 主要贡献

1. **安全性增强**: 通过强化学习技术提升 VLM 的安全性，减少有害输出。

2. **推理增强**: 增强模型在推理过程中识别和拒绝潜在有害请求的能力。

3. **多模态安全**: 针对视觉-语言模态的安全防护机制。

### 技术细节

- **强化学习**: 使用强化学习框架训练安全策略
- **奖励设计**: 专门设计的安全奖励信号
- **推理链**: 构建安全的推理链避免有害输出

## 代码结构

```
code/
├── models/          # 模型架构
├── training/        # 训练代码
├── safety/          # 安全评估
└── README.md        # 原始 README
```

## 使用方法

请参考原始 GitHub 仓库中的说明。

## 注意事项

本代码为官方开源实现。
