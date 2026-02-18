# GRIT: Teaching MLLMs to Think with Images

## 论文信息

- **标题**: GRIT: Teaching MLLMs to Think with Images
- **作者**: Eric AI Lab
- **会议**: NeurIPS 2025
- **GitHub**: [eric-ai-lab/GRIT](https://github.com/eric-ai-lab/GRIT)

## 核心内容

本文提出了一种名为 **GRIT** 的方法，旨在教多模态大型语言模型 (MLLMs) 如何利用图像进行思考和推理。

### 主要贡献

1. **视觉思维训练**: 开发了一种训练范式，使模型能够像人类一样利用图像进行中间推理。

2. **思维链增强**: 将视觉信息整合到思维链 (Chain-of-Thought) 推理过程中。

3. **空间推理能力**: 显着提升了模型在需要空间理解和视觉推理任务上的表现。

### 技术细节

- **视觉-语言对齐**: 深度融合视觉特征和语言表示
- **中间视觉表征**: 生成用于推理的中间视觉表征
- **多模态思维链**: 结合图像和文本的推理路径

## 代码结构

```
code/
├── models/          # 模型架构
├── training/        # 训练代码
├── evaluation/      # 评估代码
└── README.md        # 原始 README
```

## 使用方法

请参考原始 GitHub 仓库中的说明。

## 注意事项

本代码为官方开源实现。
