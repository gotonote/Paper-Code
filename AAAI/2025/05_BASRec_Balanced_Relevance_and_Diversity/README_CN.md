# BASRec: 平衡相关性与多样性的序列推荐

## 论文信息

- **论文标题**: Augmenting Sequential Recommendation with Balanced Relevance and Diversity
- **会议**: AAAI 2025
- **作者**: Yizhou Dang, Jiahui Zhang, Yuting Yang, Enneng Yang, Yuliang Liang, Guibing Guo, Jianzhe Zhao, Xingwei Wang
- **GitHub**: https://github.com/KingGugu/BASRec
- **论文**: https://arxiv.org/abs/2412.08300

## 核心内容

本论文提出了一种在序列推荐中平衡相关性与多样性的新方法 BASRec。

### 主要贡献

1. **相关性-多样性平衡框架**: 在保持推荐相关性的同时提升结果多样性
2. **两阶段训练策略**: 先训练基础推荐模型，再进行多样性增强
3. **在多个基准数据集上验证有效性**

### 技术细节

- **基础模型**: 支持 GRU4Rec 和 SASRec
- **数据集**: Beauty, Yelp, Home, Sports_and_Outdoors

## 代码结构

```
05_BASRec_Balanced_Relevance_and_Diversity/
├── src/
│   ├── GRU4Rec/      # GRU4Rec 实现
│   ├── SASRec/      # SASRec 实现
│   └── output/      # 日志文件
└── README.md        # 英文原版说明
```

## 使用方法

### GRU4Rec 训练

```bash
python main.py --data_name=Beauty --load_pretrain --model_idx=1 \
  --dropout_prob=0.2 --rate_min=0.2 --rate_max=0.51
```

### SASRec 训练

```bash
python main.py --data_name=Beauty --model_idx=1 --load_pretrain \
  --beta=0.4 --attention_probs_dropout_prob=0.1 --hidden_dropout_prob=0.1 \
  --n_pairs=2 --n_whole_level=3 --rate_min=0.2 --rate_max=0.71
```

## 实验数据

在以下数据集上验证:
- Beauty (亚马逊美妆)
- Yelp (餐厅点评)
- Home (家居用品)
- Sports_and_Outdoors (运动户外)

---

**来源**: 官方代码 (Official Code)
