# MoA：混合智能体增强大型语言模型能力

## 论文概述

Mixture of Agents (MoA) 是一种新颖的方法，利用多个 LLM 的集体力量来增强性能，在 AlpacaEval 2.0 上取得了最先进的结果。

## 核心内容

### 主要贡献

1. **分层架构**：采用分层架构，每层包含多个 LLM 智能体

2. **卓越性能**：在 AlpacaEval 2.0 上，MoA 使用仅开源模型就超越了 GPT-4 Omni 的 57.5%，达到了 65.1%

3. **无需微调**：通过智能体协作而非模型微调来提升性能

### 技术特点

- **多层协作**：通过多层智能体逐步优化答案
- **聚合机制**：每一层的输出作为下一层的输入
- **灵活配置**：可配置层数和每层的智能体数量

## 代码说明

本代码为官方开源代码，克隆自：https://github.com/togethercomputer/moa

### 快速开始

1. 安装 Together Python 库：`pip install together`
2. 获取 API Key：https://api.together.xyz/settings/api-keys
3. 运行示例：

```python
# 简单示例（2层，4个LLM）
python moa.py

# 高级示例（3+层）
python advanced-moa.py
```

### 环境配置

```bash
export TOGETHER_API_KEY=你的API密钥
```

## 相关链接

- 论文：https://arxiv.org/abs/2406.04692
- Discord：https://discord.com/invite/9Rk6sSeWEG

---

*来源：ICLR 2025 论文*
