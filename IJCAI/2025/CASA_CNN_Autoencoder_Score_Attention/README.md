# CASA: 基于 CNN 自编码器的分数注意力用于高效多变量长期时间序列预测

## 核心内容

CASA 提出了一种基于 CNN 自编码器的分数注意力机制，用于高效的多变量长期时间序列预测。

### 主要贡献

1. **创新架构**：提出 CNN 自编码器为基础的分数注意力机制

2. **核心技术**：
   - **通道级 Tokenization**：改进通道级特征提取
   - **模型无关特征**：具有计算效率优势

3. **优势**：
   - 高效的多变量时间序列预测
   - 较低的计算开销
   - 良好的泛化能力

## 论文信息

- **标题**: CASA: CNN Autoencoder-based Score Attention for Efficient Multivariate Long-term Time-series Forecasting
- **会议**: IJCAI 2025
- **arXiv**: https://arxiv.org/abs/2505.02011
- **作者**: Minhyuk Lee, HyeKyung Yoon, MyungJoo Kang

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 训练和评估

```bash
# ECL 数据集示例
bash ./scripts/long_term_forecast/ECL_script/CASA.sh
```

## 项目结构

```
CASA/
├── data_provider/      # 数据提供器
├── exp/               # 实验代码
├── figures/           # 图表
├── layers/            # 网络层
├── models/            # 模型定义
├── scripts/           # 训练脚本
├── utils/             # 工具函数
├── run.py             # 主运行脚本
└── requirements.txt    # 依赖
```

---

**来源**: 官方代码 - https://github.com/lmh9507/CASA
