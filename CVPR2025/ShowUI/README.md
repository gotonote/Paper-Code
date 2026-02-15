# ShowUI: GUI智能体的视觉-语言-动作模型 (CVPR 2025)

## 论文信息

- **标题**: ShowUI: One Vision-Language-Action Model for GUI Visual Agent
- **作者**: Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Weixian Lei, Lijuan Wang, Mike Zheng Shou
- **机构**: National University of Singapore, Microsoft
- **会议**: CVPR 2025
- **论文链接**: https://arxiv.org/abs/2411.17465
- **项目主页**: https://github.com/showlab/ShowUI

## 核心贡献总结

1. **开源**: 完全开源的端到端视觉-语言-动作模型
2. **轻量级**: 专为GUI智能体设计的轻量级模型
3. **多任务支持**: 支持GUI导航、计算机操作等多种任务
4. **高性能**: 在GUI智能体任务上表现出色

## 方法概述

ShowUI是一种用于GUI智能体和计算机操作的端到端视觉-语言-动作模型：

1. **VLA架构**: 视觉-语言-动作统一架构
2. **GUI理解**: 专门针对GUI界面的理解和交互
3. **动作生成**: 能够生成点击、输入等动作
4. **多数据集支持**: 支持多种GUI数据集

### 关键技术点

- **Vision-Language-Action**: 视觉-语言-动作模型
- **GUI Agent**: GUI智能体
- **Computer Use**: 计算机使用能力
- **Multi-task**: 多任务支持

## 代码结构说明

```
ShowUI/
├── model/                    # 模型定义
├── main/                     # 主程序
├── data/                     # 数据处理
├── utils/                    # 工具函数
├── prepare/                 # 数据准备
├── examples/                 # 示例
├── train.py                  # 训练脚本
├── inference_vllm.ipynb     # VLLM推理notebook
├── test.ipynb               # 测试notebook
└── README_en.md            # 英文原版README
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 推理示例

```python
from showui import ShowUI

# 加载模型
model = ShowUI.from_pretrained("showlab/ShowUI-2B")
model.eval()

# GUI交互
screenshot = load_screenshot("gui.png")
action = model(screenshot, instruction="click login")
```

## 预训练模型

| 模型 | 参数量 | 下载 |
|:-----|:------:|:-----|
| ShowUI-2B | 2B | [HuggingFace](https://huggingface.co/showlab/ShowUI-2B) |

## 数据集

| 数据集 | 描述 |
|:-------|:-----|
| ShowUI-desktop-8K | 桌面GUI数据集 |
| ShowUI-web | Web GUI数据集 |

## 来源

**来源：官方开源代码** - https://github.com/showlab/ShowUI

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
