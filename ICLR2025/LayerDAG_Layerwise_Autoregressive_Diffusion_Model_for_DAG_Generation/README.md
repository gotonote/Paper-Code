# LayerDAG：有向无环图生成的逐层自回归扩散模型

## 论文概述

LayerDAG 是一种用于有向无环图（DAG）生成的新型逐层自回归扩散模型。

## 核心内容

### 主要贡献

1. **逐层自回归扩散**：提出了一种新的图生成方法，将扩散模型与自回归机制结合

2. **DAG 约束**：确保生成的图始终满足有向无环图的约束条件

3. **高效采样**：相比传统方法，LayerDAG 能够更高效地生成分子图、流程图等复杂 DAG 结构

### 技术特点

- **逐层生成**：按层次逐步构建图结构
- **自回归机制**：每一步的生成依赖于前一步的结果
- **高质量生成分子**：能够生成分子图、程序流程图等

## 代码说明

本代码为官方开源代码，克隆自：https://github.com/Graph-COM/LayerDAG

### 安装

```bash
conda create -n LayerDAG python=3.10 -y
conda activate LayerDAG
pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
conda install -c conda-forge cudatoolkit=11.6
conda clean --all -y
pip install dgl==1.1.0+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
pip install tqdm einops wandb pydantic pandas
pip install numpy==1.26.3
```

### 训练与采样

```bash
# 训练
python train.py --config_file configs/LayerDAG/tpu_tile.yaml

# 采样和评估
python sample.py --model_path X
```

## 相关链接

- 论文：https://arxiv.org/abs/2411.02322

---

*来源：ICLR 2025 论文*
