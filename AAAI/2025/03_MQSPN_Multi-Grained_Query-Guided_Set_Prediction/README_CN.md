# MQSPN: 多粒度查询引导的集合预测网络

## 论文信息

- **论文标题**: Multi-Grained Query-Guided Set Prediction Network for Grounded Multimodal Named Entity Recognition
- **会议**: AAAI 2025
- **作者**: Jielong Tang, Zhenxing Fu, Ziyang Gong, Jianxing Yu, Xiangwei Yin, Jian Yin
- **GitHub**: https://github.com/tangjielong928/mqspn
- **论文**: https://ojs.aaai.org/index.php/AAAI/article/view/34711

## 核心内容

本论文提出了一种用于接地多模态命名实体识别（GMNER）的新型多粒度查询引导集合预测网络（MQSPN）。

### 主要贡献

1. **多粒度查询机制**: 设计了不同粒度的查询向量来引导模型学习实体表示
2. **集合预测框架**: 将任务建模为集合预测问题，避免了传统序列标注的误差传播
3. **在 Twitter-GMNER 和 Twitter-FMNERG 数据集上取得 SOTA 性能**

### 技术细节

- **模型架构**: 基于 DETR 的端到端目标检测与识别框架
- **特征提取**: 使用 VinVL 提取视觉特征
- **数据集**: Twitter15, Twitter17 等社交媒体数据集

## 代码结构

```
03_MQSPN_Multi-Grained_Query-Guided_Set_Prediction/
├── assets/           # 模型架构图
├── configs/          # 配置文件
├── data/             # 数据目录
├── run.py           # 训练/评估入口
└── README.md        # 英文原版说明
```

## 使用方法

### 环境配置

```bash
conda create -n mqspn python=3.8
conda activate mqspn
pip install -r requirements.txt
```

### 数据准备

1. 下载 Twitter-GMNER 和 Twitter-FMNERG 数据集
2. 下载推文关联图片
3. 使用 VinVL 提取候选区域特征

### 训练

```bash
python run.py train --config configs/twitter.conf
```

### 评估

```bash
python run.py eval --config configs/twitter_eval.conf
```

## 依赖

- Python 3.8
- PyTorch 1.9.1
- Transformers 4.11.3
- 建议 GPU 显存 >= 24GB

---

**来源**: 官方代码 (Official Code)
