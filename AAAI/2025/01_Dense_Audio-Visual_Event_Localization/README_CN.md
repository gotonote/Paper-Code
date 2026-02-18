# CCNet: 密集视听事件定位

## 论文信息

- **论文标题**: Dense Audio-Visual Event Localization under Cross-Modal Consistency and Multi-Temporal Granularity Collaboration
- **会议**: AAAI 2025
- **作者**: Ziheng Zhou, Jinxing Zhou, Wei Qian, Shengeng Tang, Xiaojun Chang, Dan Guo
- **GitHub**: https://github.com/zzhhfut/CCNet-AAAI2025

## 核心内容

本论文提出了一种用于密集视听事件定位的跨模态一致性+多时间粒度协作网络（CCNet）。

### 主要贡献

1. **跨模态一致性模块**：通过音频和视觉特征之间的交互学习，建立模态间的对应关系
2. **多时间粒度协作机制**：融合不同时间尺度的特征，实现精确的事件边界检测
3. **在 UnAV-100 数据集上达到 SOTA 性能**

### 技术细节

- **模型架构**: 基于 Transformer 的编码器-解码器结构
- **特征提取**: 使用 ONE-PEACE 模型提取视听特征
- **数据集**: UnAV-100 视听事件定位数据集

## 代码结构

```
01_Dense_Audio-Visual_Event_Localization/
├── configs/          # 配置文件
├── libs/            # 工具库
├── train.py         # 训练脚本
├── eval.py          # 评估脚本
└── README.md        # 英文原版说明
```

## 使用方法

### 环境配置

```bash
conda create -n ccnet python=3.8
conda activate ccnet
pip install -r requirements.txt
```

### 数据准备

从 [UnAV](https://github.com/ttgeng233/UnAV) 下载视听特征和标注，或使用百度网盘链接。

### 训练

```bash
python ./train.py ./configs/CCNet_unav100.yaml --output reproduce
```

### 评估

```bash
python ./eval.py ./configs/CCNet_unav100.yaml ./ckpt/CCNet_unav100_reproduce
```

## 依赖

- PyTorch
- transformers
- numpy
- opencv-python

---

**来源**: 官方代码 (Official Code)
