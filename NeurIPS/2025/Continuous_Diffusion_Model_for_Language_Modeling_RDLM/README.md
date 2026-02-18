# Continuous Diffusion Model for Language Modeling (RDLM)

## 论文信息

- **标题**: Continuous Diffusion Model for Language Modeling
- **作者**: Harry Jo 等
- **会议**: NeurIPS 2025
- **GitHub**: [harryjo97/RDLM](https://github.com/harryjo97/RDLM)

## 核心内容

本文提出了一种名为 **RDLM (Reverse Diffusion Language Model)** 的连续扩散模型用于语言建模。

### 主要贡献

1. **连续扩散**: 提出连续时间扩散模型处理离散文本数据。

2. **高效采样**: 开发了新的采样策略以加速生成过程。

3. **灵活架构**: 支持多种扩散调度和模型架构。

### 技术细节

- **扩散过程**: 连续的噪声添加和去除过程
- **训练目标**: 简化的训练目标函数
- **条件生成**: 支持条件和无条件文本生成

## 代码结构

```
code/
├── models/          # 模型架构
├── training/       # 训练代码
├── sampling/       # 采样策略
└── README.md       # 原始 README
```

## 使用方法

请参考原始 GitHub 仓库中的说明。

## 注意事项

本代码为官方开源实现。
