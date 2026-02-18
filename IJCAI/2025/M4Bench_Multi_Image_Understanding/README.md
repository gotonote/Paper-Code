# M4Bench: 多图像理解基准 (IJCAI 2025)

## 论文信息

- **标题**: M4Bench: Multi-Image Understanding via Multi-Domain Multi-Granularity Comparison
- **作者**: -
- **机构**: Zhejiang University
- **会议**: IJCAI 2025
- **论文链接**: -
- **数据集链接**: https://huggingface.co/datasets/Anonymous8976/M4Bench

## 核心贡献总结

1. **多图像基准**: 提出M4Bench基准，增强多图像对齐和区分能力
2. **多域多粒度比较**: 涵盖粗粒度和细粒度的单域和多域图像比较任务
3. **综合评估**: 在13个多模态大语言模型上进行全面评估
4. **新视角**: 提供多图像理解研究的多个观察和观点

## 方法概述

M4Bench针对复杂关联场景分析的需求，提出以下创新：

1. **五类比较任务**: 设计粗粒度和细粒度的单域和多域图像比较任务
2. **自动化构建管道**: 自动构建大规模多图像理解数据集
3. **多模型评估**: 评估13个主流多模态大语言模型

### 关键技术点

- **Multi-Image Understanding**: 多图像理解
- **Multi-Modal LLM**: 多模态大语言模型
- **Benchmark**: 基准测试
- **Multi-Granularity**: 多粒度

## 代码结构说明

```
M4Bench/
├── assets/                  # 图像资源
├── main.py                # 主评估脚本
├── requirements.txt       # 依赖
└── README.md             # 英文原版README
```

## 快速开始

### 环境安装

```bash
conda create -n m4bench python=3.10 -y
cd M4Bench
pip install -r requirements.txt
```

### 下载模型和数据集

```bash
mkdir llm  # 下载模型放入此目录
mkdir dataset  # 下载数据集放入此目录
```

### 运行评估

```bash
python main.py \
 --model_name Qwen2-VL-7B-Instruct \
 --task_list task1,task2
```

## 支持的模型

| 模型 | 多图像支持 | 定位支持 |
|:-----|:----------:|:--------:|
| Qwen2-VL-2B-Instruct | ✅ | ✅ |
| Qwen2-VL-7B-Instruct | ✅ | ✅ |
| InternVL2-4B | ✅ | ✅ |
| InternVL2-8B | ✅ | ✅ |
| InternVL2.5-4B | ✅ | ✅ |
| InternVL2.5-8B | ✅ | ✅ |

## 来源

**来源：官方开源代码** - https://github.com/eaglelab-zju/M4Bench

本代码直接克隆自官方仓库，仅添加了中文README文档以便于理解。
