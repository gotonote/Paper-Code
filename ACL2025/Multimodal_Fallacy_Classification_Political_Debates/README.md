# 利用上下文进行政治辩论中的多模态谬误分类 (ACL 2025 ArgMining Workshop)

## 论文信息

- **标题**: Leveraging Context for Multimodal Fallacy Classification in Political Debates
- **作者**: Alessio Pittiglio, etc.
- **机构**: -
- **会议**: ACL 2025 ArgMining Workshop
- **论文链接**: -
- **项目主页**: https://nlp-unibo.github.io/mm-argfallacy/2025/

## 核心贡献总结

1. **多模态谬误分类**: 提出针对政治辩论中多模态论证谬误的分类方法
2. **上下文信息利用**: 探索多种方式整合上下文信息提升分类性能
3. **多输入模态支持**: 支持文本、音频或文本+音频三种输入模态

## 方法概述

本文提出了一种多模态谬误分类框架：

1. **任务定义**: 论证谬误检测(AFD)和论证谬误分类(AFC)
2. **多模态融合**: 利用预训练的Transformer模型融合文本和音频信息
3. **上下文池化**: 引入ContextPool机制整合上下文信息

### 关键技术点

- **Multimodal Learning**: 多模态学习
- **Fallacy Classification**: 谬误分类
- **Context Pooling**: 上下文池化
- **Political Debate**: 政治辩论

## 代码结构说明

```
mm-argfallacy/
├── configs/                  # 配置文件
├── scripts/                 # 训练和预测脚本
├── requirements.txt         # 依赖
└── README.md               # 英文原版README
```

## 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/alessiopittiglio/mm-argfallacy.git
cd mm-argfallacy

# 创建环境
conda create -n mm_argfallacy python=3.10
conda activate mm_argfallacy

# 安装依赖
pip install -e .
pip install -r requirements.txt
```

### 训练模型

```bash
python scripts/run_training.py --config path/to/model_config.yaml
```

### 生成预测

```bash
python ./scripts/run_predict.py --config path/to/model_config.yaml --checkpoint path/to/model_checkpoint.ckpt
```

## 来源

**来源：官方开源代码** - https://github.com/alessiopittiglio/mm-argfallacy

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
