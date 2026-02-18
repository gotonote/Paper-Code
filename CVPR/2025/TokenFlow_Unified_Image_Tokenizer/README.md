# TokenFlow: Unified Image Tokenizer

## 论文信息

- **论文标题**: TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation
- **作者**: ByteVision Lab
- **机构**: ByteDance
- **论文链接**: https://arxiv.org/abs/2412.03069
- **项目主页**: https://byteflow-ai.github.io/TokenFlow/
- **HuggingFace**: https://huggingface.co/collections/ByteVisionLab/tokenflow-674ebd443b616dff1a634178
- **代码仓库**: https://github.com/ByteFlow-AI/TokenFlow

## 核心贡献总结

1. **统一图像分词器**: 首次实现同时支持多模态理解和图像生成的统一图像分词器。

2. **双码本架构**: 创新性地引入双码本架构，解耦语义级和像素级特征学习，同时通过共享映射机制保持对齐。

3. **性能卓越**: 在多模态理解任务上超越LLaVA-1.5和EMU3等旗舰模型，在文本到图像生成上与SDXL相当。

## 方法概述

TokenFlow的核心技术：

- **双码本设计**: 分离语义特征和像素级特征的学习
- **共享映射机制**: 保持不同粒度特征之间的对齐
- **统一表示**: 单一分词器支持理解和生成任务

## 代码结构说明

```
TokenFlow_Unified_Image_Tokenizer/
├── model.py              # 核心分词器代码
├── requirements.txt
├── GETTING_STARTED.md
└── README.md
```

## 运行方式

### 环境安装
```bash
pip install -r requirements.txt
```

### 下载预训练模型
```bash
# 从HuggingFace下载
# TokenFlow tokenizer
# TokenFlow-t2i (文生图)
# TokenFlow-llava-qwen2.5-14B (多模态理解)
```

### 推理
```bash
# 文本到图像
python scripts/t2i_inference.py

# 多模态理解
python scripts/llava_inference.py
```

## 依赖

- PyTorch
- transformers
- diffusers
- accelerate
- einops

## 许可证

研究用途
