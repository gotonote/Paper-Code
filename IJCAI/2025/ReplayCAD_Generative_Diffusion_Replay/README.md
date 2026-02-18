# ReplayCAD: 用于持续异常检测的生成式扩散回放

## 核心内容

ReplayCAD 提出了一种用于**持续异常检测**的生成式扩散回放方法，解决了在不断变化的数据分布下进行异常检测的挑战。

### 主要贡献

1. **新任务定义**：首次提出持续异常检测（Continual Anomaly Detection）任务

2. **两阶段框架**：
   - **阶段1**：特征引导的数据压缩 - 将每个类的数据压缩为语义和空间特征
   - **阶段2**：回放增强的异常检测 - 使用压缩的语义和空间特征回放历史类数据

3. **创新方法**：利用扩散模型生成高质量的回放数据，保持对历史异常模式的记忆

### 技术细节

- 使用预训练的扩散模型（LDM 和 Stable Diffusion）进行特征提取和回放
- 通过语义和空间特征压缩实现高效的数据回放
- 在 MVTec-AD 和 VisA 数据集上验证了方法的有效性

## 论文信息

- **标题**: ReplayCAD: Generative Diffusion Replay for Continual Anomaly Detection
- **会议**: IJCAI 2025
- **arXiv**: https://arxiv.org/abs/2505.06603
- **作者**: Lei Hu, Zhiyong Gan, Ling Deng, etc.

## 安装

```bash
conda create -n ReplayCAD python==3.8.19
conda activate ReplayCAD
pip install -r requirements.txt
```

## 数据准备

### MVTec AD
```bash
# 下载并解压 MVTec-AD 数据集，放入 data/mvtec/
```

### VisA
```bash
# 下载并解压 VisA 数据集，放入 data/visa/
```

## 训练

### 阶段1: 特征引导的数据压缩
```bash
bash textual_inversion-main/tran_visa.sh
bash textual_inversion-main/tran_mvtec.sh
```

### 阶段2: 回放增强的异常检测
```bash
# 生成回放数据
bash textual_inversion-main/generate_visa.sh
bash textual_inversion-main/generate_mvtec.sh

# 复制生成的数据
cp -r textual_inversion-main/output/visa/generate data/visa
cp -r textual_inversion-main/output/mvtec/generate data/mvtec

# 训练异常检测模型
bash run_visa.sh
bash run_mvtec.sh
```

## 预训练模型

- MVTec: 使用 LDM 预训练模型
- VisA: 使用 Stable Diffusion v1.5

---

**来源**: 官方代码 - https://github.com/HULEI7/ReplayCAD
