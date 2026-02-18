# JudgeLM: 微调大型语言模型作为可扩展评判器

## 论文概述

JudgeLM 是一种新型的 LLM 评估方法，通过微调大型语言模型作为可扩展的评判器来解决开放域评估的难题。

## 核心内容

### 主要贡献

1. **高质量评估数据集**：构建了包含任务种子、LLM 生成答案和 GPT-4 生成评判的大规模、高质量数据集

2. **新型基准测试**：提出了用于评估评判器的新基准测试

3. **多规模模型**：训练了 7B、13B、33B 不同规模的 JudgeLM 模型

4. **偏差分析**：分析了微调评判器时的关键偏差，包括：
   - 位置偏差 (Position Bias)
   - 知识偏差 (Knowledge Bias)  
   - 格式偏差 (Format Bias)

5. **技术方案**：引入了多种技术来提升评判器性能：
   - 交换增强 (Swap Augmentation)
   - 参考支持 (Reference Support)
   - 参考丢弃 (Reference Drop)

### 实验结果

- 在 PandaLM 基准测试和新的基准测试上取得了最先进的评判性能
- JudgeLM-7B 仅需 3 分钟即可评判 5K 样本（使用 8 张 A100 GPU）
- 与教师评判器的一致性超过 90%，甚至超越人与人之间的一致性
- 展示了在单答案评判、多模态模型、多答案评判和多轮对话等场景的扩展能力

## 代码说明

本代码为官方开源代码，克隆自：https://github.com/baaivision/JudgeLM

### 目录结构

- `train/` - 训练代码
- `eval/` - 评估代码
- `data/` - 数据处理脚本

## 相关链接

- 论文：https://arxiv.org/abs/2310.17631
- OpenReview：https://openreview.net/forum?id=xsELpEPn4A
- HuggingFace 模型：https://huggingface.co/BAAI/JudgeLM-7B-v1.0

---

*来源：ICLR 2025 Spotlight 论文*
