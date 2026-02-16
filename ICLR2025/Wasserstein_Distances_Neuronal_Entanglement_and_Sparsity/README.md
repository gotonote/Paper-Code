# Wasserstein距离、神经元纠缠与稀疏性

## 论文概述

本文研究了神经网络中稀疏表示的理论性质，提出了 Sparse Expansion 方法来分析困惑度（perplexity）。

## 核心内容

### 主要贡献

1. **理论分析**：使用 Wasserstein 距离分析神经网络的稀疏表示

2. **神经元纠缠**：揭示了神经元之间的纠缠现象对模型性能的影响

3. **Sparse Expansion**：提出了一种新的稀疏扩展方法

### 技术特点

- **FFN 块中的单次专家创建**：一种高效的稀疏化技术
- **困惑度评估**：提供了完整的困惑度评估框架
- **理论保证**：给出了稀疏表示的理论分析

## 代码说明

本代码为官方开源代码，克隆自：https://github.com/Shavit-Lab/Sparse-Expansion

### 环境配置

```bash
# 使用 Micromamba 创建环境
git clone https://github.com/Shavit-Lab/Sparse-Expansion.git
cd Sparse-Expansion
micromamba create -n sparse-expansion python=3.10
micromamba activate sparse-expansion
pip install -r requirements.txt
pip install -e .
```

### 环境要求

- python==3.10
- cuda==11.8
- torch==2.2.2
- transformers==4.40.2
- cudf-cu11==23.12.*
- cuml-cu11==23.12.*

## 相关链接

- 论文：https://openreview.net/forum?id=cnKhHxN3xj

---

*来源：ICLR 2025 Spotlight 论文*
