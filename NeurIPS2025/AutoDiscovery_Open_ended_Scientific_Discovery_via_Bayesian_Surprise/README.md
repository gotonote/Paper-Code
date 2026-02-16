# AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise

## 论文信息

- **标题**: AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise
- **作者**: Allen AI 研究团队
- **会议**: NeurIPS 2025
- **GitHub**: [allenai/autodiscovery](https://github.com/allenai/autodiscovery)

## 核心内容

本文提出了一种名为 **AutoDiscovery** 的开放性科学发现框架，利用贝叶斯惊喜 (Bayesian Surprise) 来驱动科学发现过程。

### 主要贡献

1. **贝叶斯惊喜驱动**: 使用贝叶斯 surprise 作为发现新知识的驱动力，系统能够识别并优先探索那些能够最大程度改变信念的假设。

2. **开放性发现**: 不局限于预设的目标，能够自主探索未知领域并产生新的科学假设。

3. **迭代式验证**: 结合假设生成、实验设计和结果分析的完整流程。

### 技术细节

- **贝叶斯推断**: 使用贝叶斯模型来量化知识不确定性
- **假设空间**: 动态扩展的假设搜索空间
- **实验规划**: 智能选择能够最大化信息增益的实验

## 代码结构

```
code/
├── discovery/        # 发现算法核心
├── experiments/     # 实验设计
├── analysis/        # 结果分析
└── README.md        # 原始 README
```

## 使用方法

请参考原始 GitHub 仓库中的说明。

## 注意事项

本代码为官方开源实现。
