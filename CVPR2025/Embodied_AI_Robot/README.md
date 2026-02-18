# Embodied AI Robot - CVPR 2025

## 论文信息

- **标题**: Learning to Manipulate Rigid Objects with Language Instructions
- **作者**: DeepMind Robotics Team
- **链接**: https://robotics.transformers.github.io/
- **会议**: CVPR 2025

## 核心贡献总结

1. 提出基于 Transformer 的机器人操作策略
2. 实现了语言到动作的端到端映射
3. 在多种机器人操作任务上达到 SOTA
4. 支持零样本泛化到新物体

## 方法概述

1. **视觉编码器**: 处理 RGB 图像输入
2. **语言编码器**: 编码语言指令
3. **动作解码器**: 输出末端执行器位姿序列
4. **模仿学习**: 从演示中学习策略

## 代码结构说明

```
Embodied_AI_Robot/
├── policy.py          # 策略网络
├── trainer.py         # 训练器
├── requirements.txt
└── README.md
```
