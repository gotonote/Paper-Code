# UrBench: 多视角城市场景大模型评估基准

## 论文信息

- **论文标题**: UrBench: A Comprehensive Benchmark for Evaluating Large Multimodal Models in Multi-View Urban Scenarios
- **会议**: AAAI 2025
- **作者**: Baichuan Zhou, Haote Yang, Dairong Ye, Junyan Bai, Tianyi Yu, Jinhua Yu, Songyang Zhang, Dahua Lin, Conghui He, Weijia Li
- **GitHub**: https://github.com/opendatalab/UrBench
- **论文**: https://arxiv.org/pdf/2408.17267

## 核心内容

UrBench 是一个专门用于评估大语言多模态模型（LMMs）在城市场景中表现的综合基准。

### 主要贡献

1. **多视角数据**: 整合了街景和卫星视图数据，以及配对的跨视图数据
2. **区域级和角色级问题**: 包含多种层次的评估问题
3. **14种城市任务类型**: 涵盖城市规划、日常问题解答等多个维度
4. **对当前 SOTA LMMs 构成重大挑战**

### 评估结果

- 最佳闭源模型 GPT-4o 准确率仅 61.2%
- 最佳开源模型 VILA-1.5-40B 准确率仅 53.1%
- 表明当前模型在多视角城市场景中的泛化能力有限

## 代码结构

```
02_UrBench_Multi-View_Urban_Scenarios/
├── assets/           # 评估结果图表
├── urbench/         # 核心代码
├── README.md        # 英文原版说明
└── setup.py         # 安装脚本
```

## 使用方法

### 安装

```bash
git clone https://github.com/opendatalab/Urbench.git
cd urbench
conda create -n urbench python=3.10
conda activate urbench
pip install -e .
```

### 开始评估

```bash
python -m accelerate.commands.launch --num_processes=2 --main_process_port=10043 \
  -m lmms_eval --model=llava_hf \
  --model_args="pretrained=\"bczhou/tiny-llava-v1-hf\",device=\"cuda\"" \
  --log_samples --log_samples_suffix tinyllava \
  --tasks urbench_test_all --output_path ./logs
```

## 数据集

- 主页: https://opendatalab.github.io/UrBench/
- HuggingFace: https://huggingface.co/datasets/bczhou/UrBench

---

**来源**: 官方代码 (Official Code)
