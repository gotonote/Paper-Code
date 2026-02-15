# Multimodal Large Language Models - CVPR 2025

## 论文信息

- **标题**: Vision-Language Models for General Visual Understanding
- **作者**: LLaVA Team
- **链接**: https://llava-vl.github.io/
- **会议**: CVPR 2025

## 核心贡献总结

1. 提出端到端的多模态大语言模型架构
2. 引入视觉指令微调方法，提升模型视觉理解能力
3. 实现超越 GPT-4V 的视觉推理性能
4. 开源了完整的训练数据和模型权重

## 方法概述

1. **视觉编码器**: 使用 CLIP ViT-L 作为视觉编码器
2. **投影层**: 线性投影将视觉特征映射到语言空间
3. **大语言模型**: 使用 Vicuna-13B 作为语言模型
4. **视觉指令微调**: 通过指令微调提升视觉理解能力

## 代码结构说明

```
Multimodal_LLM/
├── model/
│   ├── llava.py           # 主模型
│   ├── vision_encoder.py  # 视觉编码器
│   └── projector.py       # 投影层
├── data/
│   └── dataset.py         # 数据集处理
├── train.py
├── inference.py
└── README.md
```

## 关键代码讲解

```python
class LLaVA(nn.Module):
    """
    Large Language and Vision Assistant
    
    端到端的多模态大语言模型
    """
    
    def __init__(
        self,
        vision_encoder_name="clip-vit-large-patch14",
        llm_name="vicuna-13b",
        mm_projector_type="linear",
        freeze_llm=False,
        freeze_vision_encoder=True
    ):
        super().__init__()
        
        # 视觉编码器
        self.vision_encoder = CLIPVisionEncoder(vision_encoder_name)
        
        # 投影层
        self.mm_projector = Projector(
            self.vision_encoder.hidden_size,
            self.llm_config.hidden_size,
            mm_projector_type
        )
        
        # 语言模型
        self.llm = LLMModel(llm_name)
        
        # 冻结部分参数
        if freeze_vision_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
    
    def forward(self, images, texts):
        """
        前向传播
        
        Args:
            images: [B, C, H, W] 图像
            texts: tokenized 文本
        
        Returns:
            logits: 语言模型预测的 logits
        """
        # 视觉编码
        image_features = self.vision_encoder(images)
        
        # 投影到语言空间
        image_embeds = self.mm_projector(image_features)
        
        # 拼接图像和文本 embeddings
        inputs = self.embed_tokens(texts.input_ids)
        inputs = torch.cat([image_embeds, inputs], dim=1)
        
        # 语言模型前向
        outputs = self.llm(inputs)
        
        return outputs.logits
```
