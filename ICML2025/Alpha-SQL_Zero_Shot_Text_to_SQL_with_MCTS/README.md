# Alpha-SQL: 基于蒙特卡洛树搜索的零样本Text-to-SQL (ICML 2025)

## 论文信息

- **标题**: Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search
- **作者**: Boyan Li, Jiayi Zhang, Ju Fan, Yanwei Xu, Chong Chen, Nan Tang, Yuyu Luo
- **机构**: HKUST
- **会议**: ICML 2025
- **论文链接**: https://arxiv.org/abs/2502.17248
- **项目主页**: https://alpha-sql-hkust.github.io/

## 核心贡献总结

1. **MCTS框架**: 创新性地将蒙特卡洛树搜索(MCTS)应用于零样本Text-to-SQL任务
2. **LLM-as-Action-Model**: 引入LLM作为动作模型，在MCTS过程中动态生成SQL构建动作
3. **自监督奖励函数**: 提出自监督奖励函数来评估候选SQL查询的质量
4. **无需微调**: 充分利用大语言模型的知识和推理能力，无需任务特定微调

## 方法概述

Alpha-SQL是一种新型的零样本Text-to-SQL方法，核心创新在于：

1. **蒙特卡洛树搜索**: 利用MCTS框架根据部分SQL查询状态迭代推断SQL构建动作
2. **动态动作生成**: LLM-as-Action-Model在MCTS过程中动态生成SQL构建动作，引导搜索朝向更有前景的SQL查询
3. **自监督奖励**: 使用自监督奖励函数评估候选SQL查询的质量，确保更准确和高效的查询生成

### 关键技术点

- **MCTS (Monte Carlo Tree Search)**: 蒙特卡洛树搜索
- **LLM-as-Action-Model**: 大语言模型作为动作模型
- **Self-supervised Reward**: 自监督奖励函数
- **Zero-shot Learning**: 零样本学习

## 代码结构说明

```
AlphaSQL/
├── data/                    # 数据集目录
│   └── bird/               # BIRD数据集
├── config/                  # 配置文件
├── script/                  # 运行脚本
├── alphasql/               # 核心代码
│   ├── runner/            # 运行器
│   ├── templates/          # 提示模板
│   ├── database/           # 数据库处理
│   ├── llm_call/          # LLM调用
│   └── algorithm/         # MCTS算法
└── README.md              # 英文原版README
```

## 快速开始

### 环境安装

```bash
# 创建conda环境
conda create -n alphasql python=3.11
conda activate alphasql

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

下载BIRD数据集并解压到data/bird目录。

### 运行示例

```bash
# 数据预处理
bash script/preprocess.sh

# 生成SQL候选
bash script/qwen32b_bird_dev_exp.sh

# 选择最终SQL
bash script/sql_selection.sh
```

## 模型性能

| 模型 | BIRD Dev Accuracy |
|:-----|:----------------:|
| Alpha-SQL (Qwen2.5-Coder-32B) | SOTA |

## 来源

**来源：官方开源代码** - https://github.com/HKUSTDial/Alpha-SQL

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
