# Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models

## 论文信息

- **论文标题**: Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models
- **作者**: Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, Huan Ling
- **机构**: NVIDIA, University of Toronto
- **论文链接**: https://arxiv.org/abs/2503.01774
- **项目主页**: https://research.nvidia.com/labs/toronto-ai/difix3d/
- **HuggingFace**: https://huggingface.co/nvidia/difix
- **代码仓库**: https://github.com/nv-tlabs/Difix3D
- **奖项**: CVPR 2025 Oral

## 核心贡献总结

1. **单步扩散模型**: 提出使用单步扩散模型去除3D重建中的伪影。

2. **高质量3D重建**: 显著提升3D重建质量，去除各种退化伪影。

3. **高效推理**: 单步推理即可实现高质量结果，效率极高。

4. **可选参考图像引导**: 支持使用参考图像引导去噪过程。

## 方法概述

Difix3D+的核心技术：

- **单步扩散**: 使用单步扩散模型进行快速去噪
- **3D场景修复**: 专门针对3D重建的伪影去除
- **参考引导**: 可选地使用参考图像引导生成过程

## 代码结构说明

```
Difix3D_Improving_3D_Reconstructions/
├── pipeline_difix.py           # 扩散管道
├── models/                     # 模型定义
├── assets/                     # 示例资源
├── requirements.txt            # 依赖
└── README.md
```

## 运行方式

### 环境安装
```bash
pip install -r requirements.txt
```

### 快速推理
```python
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

# 加载模型
pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

# 输入图像
input_image = load_image("example_input.png")
prompt = "remove degradation"

# 推理 - 单步！
output_image = pipe(
    prompt, 
    image=input_image, 
    num_inference_steps=1, 
    timesteps    guidance_scale==[199], 
0.0
).images[0]

output_image.save("example_output.png")
```

### 使用参考图像引导
```python
# 使用参考图像引导
pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)

input_image = load_image("input.png")
ref_image = load_image("ref.png")

output_image = pipe(
    prompt, 
    image=input_image, 
    ref_image=ref_image,
    num_inference_steps=1, 
    timesteps=[199], 
    guidance_scale=0.0
).images[0]
```

## 依赖

- PyTorch
- diffusers
- transformers
- accelerate
- pillow
- numpy

## 许可证

NVIDIA研究许可证
