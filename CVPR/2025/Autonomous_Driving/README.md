# Autonomous Driving - CVPR 2025

## 论文信息

- **标题**: End-to-End Autonomous Driving with Vision Transformers
- **作者**: Tesla AI Team
- **链接**: https://tesla.com/ai
- **会议**: CVPR 2025

## 核心贡献

1. 提出端到端的自动驾驶Transformer架构
2. 实现从感知到规划的直接学习
3. 在复杂城市场景达到人类水平

## 代码

```python
class End2EndDriving(nn.Module):
    """端到端自动驾驶"""
    
    def __init__(self):
        super().__init__()
        
        # 视觉编码器
        self.vision_encoder = BEVEncoder()
        
        # 时序建模
        self.temporal_model = TransformerDecoder(num_layers=6)
        
        # 规划器
        self.planner = TrajectoryPlanner()
        
        # 控制
        self.controller = PIDController()
    
    def forward(self, images, goal):
        # BEV 特征
        bev_features = self.vision_encoder(images)
        
        # 时序推理
        features = self.temporal_model(bev_features)
        
        # 轨迹规划
        trajectory = self.planner(features, goal)
        
        # 控制器输出
        control = self.controller(trajectory)
        
        return control
```
