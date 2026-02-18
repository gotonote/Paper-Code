# Navigation World Models (NWM)

## 论文信息

- **论文标题**: Navigation World Models
- **作者**: Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun
- **机构**: AI at Meta, UC Berkeley, NYU
- **论文链接**: https://arxiv.org/abs/2412.03572
- **项目主页**: https://www.amirbar.net/nwm/
- **代码仓库**: https://github.com/facebookresearch/nwm
- **奖项**: CVPR 2025 Oral

## 核心贡献总结

1. **具身智能导航**: 提出用于具身智能导航任务的世界模型，能够预测机器人观察和动作执行结果。

2. **条件扩散Transformer (CDiT)**: 设计了高效的CDiT架构，在视觉导航任务上表现出色。

3. **大规模预训练**: 在多种机器人导航数据集上进行预训练，展现强大的零样本迁移能力。

4. **端到端学习**: 实现端到端的观察条件和动作条件的预测。

## 方法概述

NWM的核心技术包括：

- **条件扩散模型**: 使用扩散模型建模世界动态
- **Transformer骨干**: 采用Transformer架构处理时序信息
- **动作条件预测**: 根据当前观察和执行的动作预测下一状态
- **观察条件预测**: 基于观察序列预测未来观察

## 代码结构说明

```
Navigation_World_Models/
├── models.py                   # 模型定义
├── diffusion/                  # 扩散模块
├── datasets.py                 # 数据集加载
├── train.py                    # 训练脚本
├── config/                     # 配置文件
│   └── nwm_cdit_xl.yaml       # CDiT-XL配置
├── environment.yml             # 环境配置
├── interactive_model.ipynb     # 交互式Notebook
└── README.md
```

## 运行方式

### 环境安装
```bash
mamba create -n nwm python=3.10
mamba activate nwm
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip3 install decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets
```

### 数据准备
```bash
# 下载数据集（参考NoMaD项目）
# 处理数据：
python process_bags.py
python process_recon.py
```

### 训练模型
```bash
# 单机8卡训练
torchrun \
  --nnodes=1 \
  --nproc-per-node=8 \
  train.py --config config/nwm_cdit_xl.yaml
```

### 推理
```bash
python isolated_nwm_infer.py --config config/nwm_cdit_xl.yaml --checkpoint path/to/ckpt
```

## 关键代码讲解

### 1. 模型定义
```python
from models import CDiTXL

# 创建CDiT-XL模型
model = CDiTXL(
    obs_encoder=obs_encoder,
    action_encoder=action_encoder,
    diffusion_model=diffusion_model,
)
```

### 2. 扩散推理
```python
# 条件扩散采样
def sample(model, obs, actions, num_steps=50):
    # 初始化噪声
    x_t = torch.randn(obs.shape[0], 3, 320, 240)
    
    # 逐步去噪
    for t in reversed(range(num_steps)):
        with torch.no_grad():
            pred = model(x_t, obs, actions, t)
            x_t = update(x_t, pred, t)
    
    return x_t
```

## 依赖

- Python 3.10+
- PyTorch (nightly)
- transformers
- diffusers
- einops
- timm
- decord
- tqdm
- notebook
- ffmpeg

## 许可证

Meta AI研究许可证
