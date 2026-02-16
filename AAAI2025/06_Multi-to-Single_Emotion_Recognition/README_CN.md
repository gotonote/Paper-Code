# Multi-to-Single: 通过对比学习减少多模态依赖的情感识别

## 论文信息

- **论文标题**: Multi-to-Single: Reducing Multimodal Dependency in Emotion Recognition through Contrastive Learning
- **会议**: AAAI 2025
- **作者**: Arcee-LYK 等
- **GitHub**: https://github.com/Arcee-LYK/Multi-to-Single

## 核心内容

本论文提出了一种减少多模态依赖的情感识别方法，通过对比学习将多模态信息蒸馏到单模态表示中。

### 主要贡献

1. **多模态到单模态蒸馏**: 将多模态模型的知识迁移到单模态模型
2. **对比学习机制**: 通过对比学习确保单模态表示与多模态表示的一致性
3. **降低推理成本**: 推理时只需单模态输入，大幅降低计算开销

### 技术细节

- **模型架构**: 基于对比学习的多模态蒸馏框架
- **应用场景**: 多模态情感识别、对话情感分析

## 代码结构

```
06_Multi-to-Single_Emotion_Recognition/
├── M2S_demo_code/   # 演示代码
└── README.md        # 英文原版说明
```

## 使用方法

### 环境配置

```bash
conda create -n m2s python=3.8
conda activate m2s
pip install -r requirements.txt
```

### 训练

详细训练步骤请参考 M2S_demo_code 目录下的代码。

---

**来源**: 官方代码 (Official Code)
