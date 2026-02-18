# PFCD: 无提示条件扩散用于多目标图像增强

## 核心内容

PFCD 提出了一种**无提示条件扩散**方法，用于多目标图像增强任务。

### 主要贡献

1. **新方法**：首次提出无需文本提示的条件扩散模型用于多目标图像增强

2. **核心技术**：
   - 条件扩散模型架构
   - 多目标图像处理管道

3. **应用**：
   - 图像增强
   - 数据增强
   - 下游任务提升

## 论文信息

- **标题**: Prompt-Free Conditional Diffusion for Multi-object Image Augmentation
- **会议**: IJCAI 2025
- **arXiv**: https://arxiv.org/abs/2507.06146
- **HuggingFace**: https://huggingface.co/0why0/PFCD
- **作者**: Haoyu Wang, Lei Zhang, Wei Wei, Chen Ding, Yanning Zhang

## 环境准备

```bash
conda create -n env_name python=3.11 -y
conda activate env_name
pip install -r requirements.txt
```

## 数据准备

1. 下载 [COCO 数据集](https://cocodataset.org/#download)
2. 解压并放入 data 文件夹
3. 修改 `dataset/coco.py` 中的 `data_root` 为数据集路径

## 训练

```bash
# 使用提供的训练脚本
bash scripts/sdxl512/train.sh
```

可从 [HuggingFace](https://huggingface.co/0why0/PFCD/tree/main/weights) 下载预训练模型权重。

## 评估

### 下游任务评估

```bash
# 生成训练集图像
bash scripts/generate/generate_train.sh

# 标注生成的图像
bash scripts/labeling/label_train.sh
bash scripts/labeling/label_train_seg.sh

# 使用 Detectron2 训练下游任务模型
```

### 生成质量评估

```bash
# 生成验证集图像
bash scripts/generate/generate_val.sh

# 评估指标：FID, Diversity Score(LPIPS), Image Quantity Score(IQS)
python utils/metrics.py
python iqs/evaluation.py
```

## Gradio Demo

```bash
python app_gradio.py --checkpoint_path path/to/ckpt
```

## 致谢

基于以下项目构建：
- Diffusers
- Grounding DINO
- Segment Anything (SAM)
- Detectron2
- YOLOv8

---

**来源**: 官方代码 - https://github.com/00why00/PFCD
