# ICM-LoRA: 上下文元 LoRA 生成

## 核心内容

ICM-LoRA 提出了一种**上下文元 LoRA 生成**方法，能够在上下文中根据任务描述动态生成适配的 LoRA 参数。

### 主要贡献

1. **新范式**：首次提出上下文元学习生成 LoRA 参数的范式

2. **核心方法**：
   - **任务向量提取**：从 LoRA 微调模型中提取任务向量
   - **CVAE 模型**：使用条件变分自编码器学习 LoRA 参数的分布
   - **LoRA 重建**：根据任务向量和条件信息重建 LoRA 参数

3. **优势**：
   - 无需针对每个新任务进行完整的微调
   - 通过上下文信息快速生成适配的 LoRA 参数
   - 支持不同秩（rank）的 LoRA 参数生成

## 论文信息

- **标题**: In-Context Meta LoRA Generation
- **会议**: IJCAI 2025
- **arXiv**: https://arxiv.org/abs/2501.17635
- **作者**: Yihua Shao, Minxi Yang, Yang Liu, etc.

## 安装

```bash
git clone https://github.com/YihuaJerry/ICM-LoRA.git
conda create -n icmlora python=3.10
conda activate icmlora
pip install -r requirements.txt
```

## 数据准备

训练数据格式（Florence2）：
```json
{
  "id": "image_0001",
  "image_path": "/path/to/images/image_0001.jpg",
  "height": 480,
  "width": 640,
  "suffix": "dog<loc_120><loc_85><loc_340><loc_250> person<loc_50><loc_30><loc_150><loc_200> car<loc_400><loc_150><loc_600><loc_300>"
}
```

## 训练流程

### 1. 训练 LoRA 参数
使用 `train_lora/train_lora_arg.py` 脚本微调大模型，生成不同秩（1-8）的 LoRA 参数。

### 2. 处理 LoRA 参数
```bash
# 格式化 LoRA 参数
python3 reformat_lora_param.py --source_path "$SOURCE_PATH" --target_path "$TARGET_PATH"

# 扁平化和归一化
python3 utils/normalizeLoraWeight_small.py --dataset_path "$TARGET_PATH"
```

### 3. 训练参数生成器
```bash
# 获取任务向量
python ICL/last_time_step_icl.py

# 训练 CVAE 模型
python3 TrainScript_CVAE.py \
 --train_data_dir ../data/xxxx/normalized_data \
 --condition_dim xxx \
 --input_dim xxxxx \
 --task_vector_path xxxxx \
 --checkpoint_dir xxx
```

## LoRA 重建

```bash
python3 utils/ReconstructLora_cvae.py \
 --train_data_dir ../data/xxxx/normalized_data \
 --condition_dim xxx \
 --input_dim xxxxx \
 --task_vector_path xxxxx \
 --cvae_checkpoint_path xxx \
 --datasetname "dog-r=8" \
 --normalized_lora_path xxx \
 --rank 8
```

## 测试

```bash
python3 test.py \
 --download_location xxx \
 --datasetname xxx \
 --generated_lora xxx \
 --rank xxx
```

---

**来源**: 官方代码 - https://github.com/YihuaJerry/ICM-LoRA
