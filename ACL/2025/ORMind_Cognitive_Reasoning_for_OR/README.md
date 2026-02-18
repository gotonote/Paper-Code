# ORMind: 运筹学认知启发式端到端推理框架 (ACL 2025 Industry Track)

## 论文信息

- **标题**: ORMind: A Cognitive-Inspired End-to-End Reasoning Framework for Operations Research
- **作者**: Zhiyuan Wang, Bokui Chen, Yinya Huang, Qingxing Cao, Ming He, Jianping Fan, Xiaodan Liang
- **机构**: -
- **会议**: ACL 2025 Industry Track
- **论文链接**: https://aclanthology.org/2025.acl-industry.10/

## 核心贡献总结

1. **认知启发式框架**: 提出受人类认知启发的运筹学端到端推理框架
2. **反事实推理**: 引入反事实推理来改进解决方案
3. **多智能体协作**: 多个专业智能体协同工作
4. **SOTA性能**: 在NL4Opt和ComplexOR数据集上达到最佳性能

## 方法概述

ORMind旨在通过以下方式改进基于LLM的优化：

1. **结构化工作流程**: 模拟人类专家的问题解决流程
2. **语义编码器**: 将自然语言问题转换为形式化表示
3. **形式化思考**: 将问题形式化为数学优化模型
4. **执行编译器**: 将形式化模型转化为可执行代码
5. **元认知监督器**: 监控和调整推理过程

### 核心技术组件

- **Semantic Encoder**: 语义编码器
- **Formalization Thinking**: 形式化思考
- **Executive Compiler**: 执行编译器
- **Metacognitive Supervisor**: 元认知监督器
- **System 2 Reasoner**: 系统2推理器

## 代码结构说明

```
ORMind/
├── run_exp.py              # NL4Opt数据集实验
├── run_exp_ComplexOR.py    # ComplexOR数据集实验
├── requirements.txt        # 依赖
└── README.md              # 英文原版README
```

## 快速开始

### 环境安装

```bash
git clone https://github.com/XiaoAI1989/ORMind.git
cd ORMind
pip install -r requirements.txt
```

### 运行实验

```bash
# NL4Opt数据集
python run_exp.py

# ComplexOR数据集
python run_exp_ComplexOR
```

## 性能表现

| 数据集 | 准确率 |
|:-------|:------:|
| NL4Opt | 68.8% |
| ComplexOR | 40.5% |

## 来源

**来源：官方开源代码** - https://github.com/XiaoAI1989/ORMind

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
