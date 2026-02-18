# T2S: 基于文本到系列扩散模型的高分辨率时间序列生成

## 核心内容

T2S 是第一个**领域无关的**（domain-agnostic）文本到时间序列生成框架，允许用户（包括非专业人员和专业人员）从自然语言描述生成高分辨率、时间语义对齐的时间序列。

### 主要贡献

1. **首个领域无关框架**：T2S 是第一个能够从自然语言生成时间序列的领域无关模型

2. **TSFragment-600K 数据集**：首个在6个经典领域对齐的片段级文本-时间序列多模态数据集

3. **核心技术组件**：
   - **T2S-DiT**: 为自然语言条件生成定制的基于扩散的Transformer
   - **LA-VAE**: 预训练的长度自适应变分自编码器，支持可变长度序列生成

### 应用场景

- **包容性数据交互**：非专家可以描述时间行为并生成合成数据
- **专业人员快速原型设计**：使用简单的文本描述快速模拟系统时间动态
- **压力测试**：模拟边缘情况（如极端需求激增）来评估系统鲁棒性

## 论文信息

- **会议**: IJCAI 2025
- **数据集**: [TSFragment-600K](https://huggingface.co/datasets/WinfredGe/TSFragment-600K)
- **预训练模型**: [T2S-LA-VAE](https://huggingface.co/WinfredGe/T2S-pretrained_LA-VAE) 和 [T2S-DiT](https://huggingface.co/WinfredGe/T2S-DiT)

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
# 注意: T2S 需要 torch==2.3.1
```

## 数据准备

```python
# 从 Hugging Face 加载数据集
from datasets import load_dataset
ds = load_dataset("WinfredGe/TSFragment-600K")
```

或下载预处理的[数据集](https://drive.google.com/file/d/1tV0xBd0ToWvuLpI5Ocd49uM3QcRkP4NT/view?usp=sharing)

## 预训练 LA-VAE

```bash
python pretrained_lavae_unified.py --dataset_name ETTh1 --save_path 'results/saved_pretrained_models/' --mix_train True
```

## 训练和推理

```bash
# 训练
python train.py --dataset_name 'ETTh1'

# 推理
python infer.py --dataset_name 'ETTh1_24' --cfg_scale 9.0 --total_step 10
python infer.py --dataset_name 'ETTh1_48' --cfg_scale 9.0 --total_step 10
python infer.py --dataset_name 'ETTh1_96' --cfg_scale 9.0 --total_step 10
```

## 评估

```bash
python evaluation.py --dataset_name 'ETTh1_24' --cfg_scale 9.0 --total_step 10
```

## 快速复现

1. 安装 Python 3.10 和依赖
2. 下载 [TSFragment-600K 数据](https://huggingface.co/datasets/WinfredGe/TSFragment-600K) 和 [T2S 检查点](https://drive.google.com/file/d/1T-gjPMvnpSFpkkUSZpAeeIqALThOQydT/view?usp=sharing)
3. 直接运行评估脚本 `./scripts_validation_only.sh`

---

**来源**: 官方代码 - https://github.com/WinfredGe/T2S
