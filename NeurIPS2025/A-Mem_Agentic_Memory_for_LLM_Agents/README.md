# A-Mem: Agentic Memory for LLM Agents

## 论文信息

- **标题**: A-Mem: Agentic Memory for LLM Agents
- **作者**: Wujiang Xu 等
- **会议**: NeurIPS 2025
- **GitHub**: [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem)

## 核心内容

本文提出了一种名为 **A-Mem (Agentic Memory)** 的新型记忆框架，用于增强大型语言模型 (LLM) 代理的能力。

### 主要贡献

1. **智能记忆架构**: 设计了一种能够动态更新和检索的记忆系统，使代理能够从过去的交互中学习和改进。

2. **经验积累机制**: 代理可以积累经验，并在后续任务中灵活运用这些经验来提高性能。

3. **上下文感知**: 记忆系统能够根据当前任务上下文智能地选择相关信息。

### 技术细节

- **记忆检索**: 使用向量相似度检索相关记忆
- **记忆更新**: 基于强化学习的记忆更新策略
- **长程依赖**: 有效处理跨多轮交互的依赖关系

## 代码结构

```
code/
├── src/              # 源代码
├── configs/          # 配置文件
├── scripts/          # 脚本文件
└── README.md         # 原始 README
```

## 使用方法

请参考原始 GitHub 仓库中的说明。

## 注意事项

本代码为官方开源实现。
