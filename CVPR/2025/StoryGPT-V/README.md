# StoryGPT-V: Large Language Models as Consistent Story Visualizers

> CVPR 2025

## 论文信息

- **标题**: StoryGPT-V: Large Language Models as Consistent Story Visualizers
- **作者**: Xiaoqian Shen, Mohamed Elhoseiny
- **链接**: [GitHub](https://github.com/xiaoqian-shen/StoryGPT-V)
- **原始论文**: [ArXiv](https://arxiv.org/abs/xxxx.xxxxx)

## 核心贡献

1. 提出利用大语言模型进行故事一致性可视化
2. 生成连贯的图像序列讲述完整故事
3. 无需额外训练即可实现故事生成

## 方法概述

StoryGPT-V结合LLM的推理能力和图像生成模型，创建具有叙事一致性的图像序列。

## 代码结构

```
StoryGPT-V/
├── llm/             # LLM接口
├── diffusion/       # 扩散模型
├── story.py         # 故事生成
└── requirements.txt
```

## 运行方式

```bash
pip install -r requirements.txt
python story.py --prompt "A cat chasing a mouse..."
```

## 依赖

- PyTorch >= 2.0
- diffusers
- transformers
