# Answer, Assemble, Ace：理解Transformer如何回答多项选择题

## 论文概述

本文深入研究了 Transformer 模型如何回答多项选择题，揭示了其内部工作机制。

## 核心内容

### 主要贡献

1. **机制分析**：深入分析了 Transformer 在多项选择任务中的工作原理

2. **答案组装过程**：揭示了模型如何"组装"答案的过程

3. **关键发现**：
   - 答案选择过程中的注意力机制作用
   - 隐藏状态如何编码答案信息
   - 上下文示例的影响

### 技术方法

- **信息追踪**：使用 nethook 工具捕获模型的隐藏状态
- **向量算术**：分析答案相关的向量表示
- **消融实验**：验证各组件的贡献

## 代码说明

本代码为官方开源代码，克隆自：https://github.com/allenai/understanding_mcqa

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 克隆 MEMIT 仓库（需要使用其中的 nethook 工具）
git clone https://github.com/kmeng01/memit.git
# 将 util/nethook.py 复制到本项目的 util/ 目录
# 或添加到 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/memit/
```

### 运行示例

```bash
python run_experiments.py \
  --in_context_examples 3 \
  --icl_labels ABB \
  --model llama2-7b \
  --dataset hellaswag
```

支持的模型：llama2-7b, llama2-13b, llama2-7b-chat, olmo-7b 等

## 相关链接

- 论文：https://arxiv.org/abs/2407.15018
- OpenReview：https://openreview.net/forum?id=6NNA0MxhCH

---

*来源：ICLR 2025 论文*
