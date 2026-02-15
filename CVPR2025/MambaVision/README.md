# MambaVision: 混合Mamba-Transformer视觉骨干网络 (CVPR 2025)

## 论文信息

- **标题**: MambaVision: A Hybrid Mamba-Transformer Vision Backbone
- **作者**: Ali Hatamizadeh, Jan Kautz
- **机构**: NVIDIA
- **会议**: CVPR 2025
- **论文链接**: https://arxiv.org/abs/2407.08083
- **项目主页**: https://github.com/NVlabs/MambaVision

## 核心贡献总结

1. **新型混合架构**: 提出混合Mamba-Transformer视觉骨干网络
2. **SOTA性能**: 在Top-1准确率和吞吐量方面达到新的Pareto最优
3. **创新Mixer块**: 引入新型mixer块，通过对称路径增强全局上下文建模
4. **分层架构**: 采用自注意力块和mixer块的分层架构

## 方法概述

MambaVision是一种混合Mamba-Transformer视觉骨干网络，核心创新在于：

1. **混合架构**: 结合Mamba和Transformer的优势
2. **新型Mixer块**: 对称路径设计，无需SSM
3. **全局上下文增强**: 增强全局上下文建模能力
4. **分层设计**: 分层架构处理不同尺度特征

### 关键技术点

- **Mamba-Transformer Hybrid**: Mamba-Transformer混合架构
- **Mixer Block**: 新型mixer块设计
- **Global Context**: 全局上下文建模
- **Hierarchical Architecture**: 分层架构

## 代码结构说明

```
MambaVision/
├── mambavision/              # 核心模型代码
├── object_detection/         # 目标检测
├── semantic_segmentation/   # 语义分割
├── setup.py                  # 安装脚本
├── Dockerfile               # Docker配置
└── README_en.md            # 英文原版README
```

## 快速开始

### 安装

```bash
pip install -e .
```

### 使用示例

```python
import torch
import timm

# 使用MambaVision模型
model = timm.create_model('mambavision_small', pretrained=True)
model.eval()

# 推理
input_tensor = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(input_tensor)
```

## 模型规模

| 模型 | 参数量 | ImageNet Top-1 |
|:-----|:------:|:--------------:|
| MambaVision-Tiny | - | - |
| MambaVision-Small | - | - |
| MambaVision-Base | - | - |
| MambaVision-Large | - | - |

## 来源

**来源：官方开源代码** - https://github.com/NVlabs/MambaVision

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
