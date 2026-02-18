# StyleGAN-Llava: Stylized Image Generation with Multi-Modal Control - CVPR 2025

## 论文信息

- **标题**: StyleGAN-Llava: Stylized Image Generation with Multi-Modal Control
- **作者**: Yuxin Zhang, Chengwei Chen et al.
- **链接**: https://github.com/styleganllava/styleganllava
- **会议**: CVPR 2025

## 核心贡献总结

1. 提出基于 GAN 的风格化图像生成方法
2. 首次实现文本和图像共同控制的风格迁移
3. 在保持内容结构的同时实现高质量风格转换

## 方法概述

1. **双分支编码器**: 分别编码内容和风格
2. **样式注入模块**: 将风格特征注入生成过程
3. **多模态融合**: 文本和图像特征融合

## 关键代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleGANLlava(nn.Module):
    """StyleGAN-Llava 风格化图像生成"""
    
    def __init__(self, latent_dim=512, style_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        
        # 内容编码器
        self.content_encoder = ContentEncoder()
        
        # 风格编码器
        self.style_encoder = StyleEncoder()
        
        # 生成器
        self.generator = StyleGenerator(latent_dim, style_dim)
    
    def forward(self, content_img, style_ref, text_prompt=None):
        # 编码内容
        content_feat = self.content_encoder(content_img)
        
        # 编码风格
        style_feat = self.style_encoder(style_ref)
        
        # 融合文本（可选）
        if text_prompt is not None:
            text_feat = self.encode_text(text_prompt)
            style_feat = self.fuse_style_text(style_feat, text_feat)
        
        # 生成
        output = self.generator(content_feat, style_feat)
        
        return output
```
