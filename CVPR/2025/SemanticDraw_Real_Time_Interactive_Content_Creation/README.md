# SemanticDraw: Real-Time Interactive Content Creation

## 论文信息

- **论文标题**: SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion Models
- **作者**: Seunghoon Hong, et al.
- **机构**: Google Research
- **论文链接**: https://arxiv.org/abs/2506.15314
- **项目主页**: https://semantic-draw.github.io/
- **代码仓库**: https://github.com/ironjr/semantic-draw

## 核心贡献总结

1. **实时交互式图像编辑**: 实现基于图像扩散模型的实时交互式内容创建，用户可以通过简单的交互进行图像编辑。

2. **语义掩码引导**: 提出语义掩码引导机制，精确控制编辑区域。

3. **快速推理优化**: 优化推理速度，实现接近实时的编辑体验。

4. **高质量结果**: 保持高质量的生成结果，同时实现实时交互。

## 方法概述

SemanticDraw的核心技术包括：

- **交互式掩码生成**: 实时生成用户意图的语义掩码
- **局部编辑**: 仅对指定区域进行编辑，保持其他区域不变
- **快速扩散**: 优化扩散过程以实现实时性能

## 代码结构说明

```
SemanticDraw_Real_Time_Interactive_Content_Creation/
├── app.py                      # Gradio应用
├── inference.py                # 推理代码
├── models/                     # 模型定义
├── utils/                      # 工具函数
├── requirements.txt            # 依赖
└── ...
```

## 运行方式

### 环境安装
```bash
cd SemanticDraw_Real_Time_Interactive_Content_Creation/code
pip install -r requirements.txt
```

### 运行演示
```bash
python app.py
```

## 依赖

- PyTorch
- diffusers
- gradio
- opencv-python
- pillow

## 许可证

研究用途
