# 动态图生成模型的质量度量

## 论文概述

本文提出了 JL-Metric，一种用于评估连续时间动态图（CTDG）生成模型的 principled 方法。

## 核心内容

### 主要贡献

1. **统一评估框架**：提出了一种原则性的方法来评估动态图生成模型的质量

2. **多维度评估**：从多个角度评估生成图的质量，包括：
   - 时间动态特性
   - 结构特性
   - 节点属性

3. **实用工具**：提供了可直接使用的度量实现

### 技术特点

- **输入格式**：接受 PyTorch Geometric 的 TemporalData 对象
- **包含信息**：源节点 (src)、目标节点 (dst)、时间戳 (t)、消息/特征 (msg)

## 代码说明

本代码为官方开源代码，克隆自：https://github.com/ryienh/jl-metric

### 安装

```bash
pip install jl-metric
```

### 基本使用

```python
from jl_metric import JLEvaluator

# 初始化评估器
evaluator = JLEvaluator()

# 准备图数据为 PyG TemporalData 对象
# reference_graph = 真实图或参考图
# generated_graph = 模型生成的图

# 创建输入字典
input_dict = {
    'reference': reference_graph,
    'generated': generated_graph
}

# 评估并获取结果
result_dict = evaluator.eval(input_dict)
print(f"JL-Metric score: {result_dict['JL-Metric']}")
```

## 相关链接

- 论文：https://openreview.net/forum?id=8bjspmAMBk

---

*来源：ICLR 2025 论文*
