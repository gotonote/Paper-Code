# Scene Understanding - CVPR 2025

## 论文信息

- **标题**: Open-Vocabulary Scene Understanding
- **作者**: Meta AI
- **链接**: https://segment-anything.com/
- **会议**: CVPR 2025

## 核心贡献

1. 首次实现开放词汇表场景理解
2. 支持任意类别的语义分割
3. 零样本泛化到新场景

## 代码

```python
class OpenVocabularySegmentation(nn.Module):
    """开放词汇表分割"""
    
    def __init__(self):
        super().__init__()
        # 使用 SAM 作为基础
        from segment_anything import sam_model_registry
        self.sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
        
        # 文本编码器
        self.text_encoder = CLIPTextEncoder()
        
        # 分割头
        self.mask_decoder = MaskDecoder()
    
    def forward(self, image, class_names):
        # 图像特征
        image_features = self.sam.image_encoder(image)
        
        # 文本特征
        text_features = self.text_encoder(class_names)
        
        # 解码分割 mask
        masks = self.mask_decoder(image_features, text_features)
        
        return masks
```
