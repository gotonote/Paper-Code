# FSTA-SNN: 基于频率的空时注意力脉冲神经网络

## 论文信息

- **论文标题**: FSTA-SNN: Frequency-based Spatial-Temporal Attention Module for Spiking Neural Networks
- **会议**: AAAI 2025
- **作者**: Yukairong 等
- **GitHub**: https://github.com/yukairong/FSTA-SNN

## 核心内容

本论文提出了一种基于频率的空时注意力模块（FSTA），用于增强脉冲神经网络（SNN）的时空建模能力。

### 主要贡献

1. **频率域注意力机制**: 将时序信号转换到频率域进行特征提取，更有效地捕捉时空依赖关系
2. **空时联合建模**: 同时建模空间维度和时间维度的特征交互
3. **生物合理性**: 脉冲神经元的发放模式与频率域分析高度契合

### 技术细节

- **模型架构**: 结合频率分析与传统 SNN 结构
- **应用场景**: 事件相机数据处理、神经形态视觉识别

## 代码结构

```
04_FSTA-SNN_Frequency-based_Spatial-Temporal_Attention/
├── config/           # 配置文件
├── data/             # 数据集
├── dvs_utils/        # DVS 事件相机工具
├── linklink/         # 脉冲神经网络工具库
├── models/           # 模型定义
├── train.py          # 训练脚本
└── README.md         # 英文原版说明
```

## 使用方法

### 环境配置

```bash
conda create -n fsta-snn python=3.x
conda activate fsta-snn
pip install -r requirements.txt
```

### 训练

```bash
python train.py --config config/your_config.yaml
```

---

**来源**: 官方代码 (Official Code)
