# GaussianCity: Unbounded 3D City Generation

## 论文信息

- **论文标题**: Generative Gaussian Splatting for Unbounded 3D City Generation
- **作者**: Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, Ziwei Liu
- **机构**: S-Lab, Nanyang Technological University
- **论文链接**: https://arxiv.org/abs/2406.06526
- **项目主页**: https://haozhexie.com/project/gaussian-city
- **HuggingFace**: https://huggingface.co/spaces/hzxie/gaussian-city
- **代码仓库**: https://github.com/hzxie/GaussianCity

## 核心贡献总结

1. **无界3D城市生成**: 首次实现无界3D城市的生成，能够生成大规模城市场景。

2. **高斯溅射表示**: 利用3D高斯溅射（Gaussian Splatting）作为场景表示，实现高质量实时渲染。

3. **模块化生成**: 分别使用背景生成器（BG-Generator）和建筑生成器（BLDG-Generator）生成场景的不同组件。

4. **实时渲染**: 基于高斯溅射技术，实现实时高质量渲染。

## 方法概述

GaussianCity的核心技术包括：

- **双阶段生成**: 先生成背景，再生成建筑物
- **3D高斯表示**: 使用3D高斯函数表示场景，实现可微渲染
- **大规模场景**: 支持无界城市场景的生成和渲染

## 代码结构说明

```
GaussianCity_Unbounded_3D_City_Generation/
├── models.py              # 核心模型代码
├── config.py              # 配置文件
├── run.py                 # 推理脚本
├── requirements.txt
└── README.md
```

## 运行方式

### 环境安装
```bash
# 安装PyTorch
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# 克隆仓库
git clone https://github.com/hzxie/GaussianCity
cd GaussianCity

# 安装依赖
pip install -r requirements.txt

# 编译CUDA扩展
cd extensions
for e in `ls -d */`; do
  cd $e
  pip install .
  cd ..
done
```

### 推理
```bash
# 下载预训练模型
# BG-Generator: output/rest.pth
# BLDG-Generator: output/bldg.pth

# 运行推理
python3 scripts/inference.py
```

### HuggingFace演示
```bash
# 在线体验
# https://huggingface.co/spaces/hzxie/gaussian-city
```

## 依赖

- Python 3.11+
- PyTorch 2.4.1+
- CUDA 11.8+
- ninja
- spconv
- plyfile
- opencv-python

## 许可证

研究用途
