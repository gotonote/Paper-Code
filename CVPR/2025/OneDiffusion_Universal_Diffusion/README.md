# OneDiffusion: Universal Diffusion

## 论文信息

- **论文标题**: One Diffusion to Generate Them All
- **作者**: Le Duong, et al.
- **机构**: Meta AI
- **论文链接**: https://arxiv.org/abs/2411.16318
- **项目主页**: https://lehduong.github.io/OneDiffusion-homepage/
- **HuggingFace**: https://huggingface.co/lehduong/OneDiffusion
- **代码仓库**: https://github.com/lehduong/OneDiffusion

## 核心贡献总结

1. **统一多任务扩散模型**: OneDiffusion是一个通用的、大规模扩散模型，无缝支持双向图像合成和理解，涵盖多种任务。

2. **多任务支持**: 支持文生图、图生图、图像编辑、主体驱动生成、图像描述、视觉问答等多种任务。

3. **高效推理**: 通过量化等技术优化，推理显存需求降至约21GB。

4. **灵活的任务切换**: 通过提示词格式自动切换不同任务模式。

## 方法概述

OneDiffusion采用统一的Diffusion Transformer架构：

- **双向生成与理解**: 统一处理图像生成和理解任务
- **任务提示格式**: 使用特殊标记（如`[[text2image]]`, `[[image2image]]`, `[[subject_driven]]`等）区分不同任务
- **大规模预训练**: 在大规模数据集上训练，具备强大的泛化能力

## 代码结构说明

```
OneDiffusion_Universal_Diffusion/
├── onediffusion/               # 核心代码
│   ├── diffusion/              # 扩散模型
│   │   └── pipelines/          # 推理管道
│   │       └── onediffusion.py
│   └── ...
├── inference.py                # 推理脚本
├── gradio_demo.py              # Gradio演示
├── PROMPT_GUIDE.md             # 提示词指南
├── requirements.txt             # 依赖
└── assets/                     # 示例资源
```

## 运行方式

### 环境安装
```bash
conda create -n onediffusion_env python=3.8
conda activate onediffusion_env
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -r requirements.txt
```

### 文生图推理
```python
import torch
from onediffusion.diffusion.pipelines.onediffusion import OneDiffusionPipeline

device = torch.device('cuda:0')
pipeline = OneDiffusionPipeline.from_pretrained("lehduong/OneDiffusion").to(
    device=device, dtype=torch.bfloat16
)

output = pipeline(
    prompt="[[text2image]] A bipedal black cat wearing a huge oversized witch hat",
    negative_prompt="monochrome, low-res, bad anatomy...",
    num_inference_steps=50,
    guidance_scale=4,
    height=1024,
    width=1024,
)
output.images[0].save('output.jpg')
```

### 主体驱动生成
```python
# 需要提供参考图像
output = pipeline(
    prompt="[[subject_driven]] A photo of a cat",
    input_image=reference_image,  # 提供主体图像
    num_inference_steps=50,
    guidance_scale=4,
)
```

### 运行Gradio演示
```bash
python gradio_demo.py --captioner molmo  # 使用Molmo作为captioner
```

## 关键代码讲解

### 1. Pipeline加载
```python
from onediffusion.diffusion.pipelines.onediffusion import OneDiffusionPipeline

# 加载预训练模型
pipeline = OneDiffusionPipeline.from_pretrained(
    "lehduong/OneDiffusion",
    torch_dtype=torch.bfloat16,
    use_safetensors=True
)
pipeline.to("cuda")
```

### 2. 不同任务提示词格式
```python
# 文生图
prompt = "[[text2image]] A beautiful landscape"

# 图像编辑
prompt = "[[image_editing]] Make the sky blue"
input_image = load_image("input.jpg")

# 主体驱动生成
prompt = "[[subject_driven]] A photo on the beach"
input_image = subject_image

# 图像描述
prompt = "[[image_captioning]]"
input_image = image_to_describe
```

### 3. 自定义推理参数
```python
output = pipeline(
    prompt=prompt,
    input_image=input_image,  # 可选
    num_inference_steps=50,   # 扩散步数
    guidance_scale=7.0,        # Classifier-free guidance强度
    height=1024,              # 输出高度
    width=1024,               # 输出宽度
    seed=42,                  # 随机种子
)
```

## 依赖

- Python 3.8+
- PyTorch 2.3.1+
- CUDA 11.8+
- transformers
- accelerate
- einops
- torchvision
- pytorch3d
- gradio
- accelerate
- pillow
- numpy

## 性能指标

| 任务 | 描述 |
|------|------|
| Text-to-Image | 1024x1024 高质量图像生成 |
| Subject-Driven | 基于参考图像的主体生成 |
| Image Editing | 指令引导的图像编辑 |
| Image Captioning | 图像描述生成 |

## 注意事项

1. 显存需求：约21GB（使用量化后的Molmo）
2. 支持多种分辨率：256, 512, 1024等
3. 提示词格式很重要，必须使用正确的任务标记

## 许可证

研究用途，详见项目页面。
