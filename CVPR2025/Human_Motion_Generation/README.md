# Human Motion Generation - CVPR 2025

## 论文信息

- **标题**: MotionGPT: Human Motion as a Foreign Language
- **作者**: Tsinghua AI Lab
- **链接**: https://motion-gpt.github.io/
- **会议**: CVPR 2025

## 核心贡献

1. 将人体运动建模为语言生成任务
2. 提出基于 Transformer 的运动生成模型
3. 支持文本到运动和运动到文本的双向生成

## 代码

```python
class MotionGPT(nn.Module):
    """人体运动生成模型"""
    
    def __init__(self, num_joints=22, hidden_dim=512):
        super().__init__()
        
        # 运动编码器
        self.motion_encoder = MotionEncoder(num_joints, hidden_dim)
        
        # Transformer
        self.transformer = Transformer(hidden_dim, num_layers=8)
        
        # 文本解码器
        self.text_decoder = nn.Linear(hidden_dim, 50000)  # vocab size
    
    def forward(self, motion, text_prompt):
        features = self.motion_encoder(motion)
        output = self.transformer(features, text_prompt)
        return self.text_decoder(output)
```
