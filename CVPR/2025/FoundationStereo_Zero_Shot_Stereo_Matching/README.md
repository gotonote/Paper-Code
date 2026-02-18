# FoundationStereo: Zero-Shot Stereo Matching

## 论文信息

- **论文标题**: FoundationStereo: Zero-Shot Stereo Matching
- **作者**: Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield
- **机构**: NVIDIA
- **论文链接**: https://arxiv.org/abs/2501.09898
- **项目主页**: https://nvlabs.github.io/FoundationStereo/
- **代码仓库**: https://github.com/NVlabs/FoundationStereo
- **奖项**: CVPR 2025 Oral (Best Paper Nomination)

## 核心贡献总结

1. **零样本泛化能力**: 首次构建了专为立体匹配设计的基础模型，在未见过真实数据的情况下实现强大的零样本泛化能力。

2. **大规模合成数据集**: 构建了包含100万对立体图像的大规模合成训练数据集，具有高度多样性和照片级真实感。

3. **Side-Tuning特征骨架**: 设计了Side-Tuning特征骨架，将视觉基础模型（ViT）的单目先验知识适配到立体匹配任务，有效缓解sim-to-real差距。

4. **长距离上下文推理**: 引入长距离上下文推理机制，有效过滤错误5. **SOTA性能**: 在Middlebury和ETH3D匹配。

等权威基准上取得第一名。

## 方法概述

FoundationStereo的核心创新包括：

- **大规模合成数据构建**: 创建了100万对立体图像的合成数据集，包含多样化的场景和光照条件
- **自动数据筛选**: 设计了自动筛选流程，去除模糊和低质量的样本
- **Side-Tuning适配器**: 利用预训练的DINOv2特征，通过Side-Tuning方式适配到立体匹配任务
- **代价体积滤波**: 使用Transformer进行长距离上下文建模，提高匹配精度
- **层次化推理**: 支持高分辨率图像的层次化推理策略

## 代码结构说明

```
FoundationStereo_Zero_Shot_Stereo_Matching/
├── models.py              # 核心模型代码
├── Utils.py               # 工具函数
├── environment.yml        # 环境配置
├── requirements.txt
└── README.md
```

## 运行方式

### 环境安装
```bash
cd FoundationStereo_Zero_Shot_Stereo_Matching/code
conda env create -f environment.yml
conda run -n foundation_stereo pip install flash-attn
conda activate foundation_stereo
```

### 下载预训练模型
```bash
# 从Google Drive下载模型权重
# 模型23-51-11: 基于ViT-large的最佳模型
# 模型11-33-40: 基于ViT-small的快速模型
# 将模型文件夹放入 ./pretrained_models/
```

### 运行演示
```bash
python scripts/run_demo.py \
    --left_file ./assets/left.png \
    --right_file ./assets/right.png \
    --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth \
    --out_dir ./test_outputs/
```

### 高分辨率推理
```bash
# 使用层次化推理获得全分辨率深度图
python scripts/run_demo.py --left_file ... --right_file ... --hiera 1

# 或使用降采样加快推理速度
python scripts/run_demo.py --left_file ... --right_file ... --scale 0.5
```

## 关键代码讲解

### 1. 模型加载
```python
import torch
from core.foundation_stereo import FoundationStereo

# 加载模型
model = FoundationStereo(args)
checkpoint = torch.load(args.ckpt_dir, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.cuda()
model.eval()
```

### 2. 立体匹配推理
```python
# 输入：左图和右图
left_img = load_image(left_path)  # (H, W, 3)
right_img = load_image(right_path)  # (H, W, 3)

# 预处理
left_input = preprocess(left_img)
right_input = preprocess(right_img)

# 推理
with torch.no_grad():
    pred_disp = model(left_input, right_input)

# pred_disp: 视差图，转换为深度: depth = baseline * focal / disparity
```

### 3. 深度/视差转换
```python
# 视差转深度
def disparity_to_depth(disparity, baseline, focal):
    depth = baseline * focal / (disparity + 1e-8)
    return depth

# 深度转点云
def depth_to_point_cloud(depth, intrinsic, extrinsic):
    # 反投影得到3D点坐标
    pass
```

### 4. TensorRT加速推理
```bash
# 导出ONNX模型
XFORMERS_DISABLED=1 python scripts/make_onnx.py \
    --save_path ./pretrained_models/foundation_stereo.onnx \
    --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth \
    --height 448 --width 672 --valid_iters 20

# 转换为TensorRT
trtexec --onnx=pretrained_models/foundation_stereo.onnx \
    --verbose --saveEngine=pretrained_models/foundation_stereo.plan --fp16
```

## 依赖

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8
- flash-attn
- einops
- timm
- opencv-python
- pycolmap (可选)

## 性能指标

在Middlebury基准上的表现:

| 方法 | D1-all | D1-bad |
|------|--------|--------|
| FoundationStereo | 2.88% | 0.89% |
| 其他方法 | ... | ... |

## 注意事项

1. 输入图像必须是校正过的立体图像对（无鱼眼畸变，极线水平对齐）
2. 不要交换左右图像
3. 建议使用无损压缩的PNG文件
4. 支持单目和红外立体图像

## 许可证

代码采用NVIDIA许可证，模型权重仅供研究使用。
