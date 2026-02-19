# QUAG: Query-centric Audio-Visual Cognition Network

> AAAI 2025

## 论文信息

- **标题**: Query-centric Audio-Visual Cognition Network for Moment Retrieval, Segmentation and Step-Captioning
- **作者**: tuyunbin
- **链接**: [GitHub](https://github.com/tuyunbin/QUAG)

## 核心贡献

1. 提出以查询为中心的视听认知网络
2. 同时处理时刻检索、分割和步骤字幕生成
3. 在多个数据集上取得SOTA性能

## 方法概述

QUAG通过跨模态注意力机制，实现了对视频中特定时刻的精准定位和理解。

## 运行方式

```bash
pip install -r requirements.txt
python main.py --query "cooking pasta"
```

## 依赖

- PyTorch >= 1.10
- librosa
- opencv-python
