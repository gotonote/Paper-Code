# CATCH: Channel-Aware Multivariate Time Series Anomaly Detection

> ICLR 2025

## 论文信息

- **标题**: CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching
- **作者**: decisionintelligence
- **链接**: [GitHub](https://github.com/decisionintelligence/CATCH)

## 核心贡献

1. 提出基于频率修补的通道感知多元时间序列异常检测
2. 充分利用通道间的相关性
3. 在多个基准数据集上取得领先性能

## 方法概述

CATCH通过频率域分析和通道感知机制，准确检测多元时间序列中的异常点。

## 运行方式

```bash
pip install -r requirements.txt
python detect.py --data swat
```

## 依赖

- PyTorch >= 1.10
- scipy
- pandas
