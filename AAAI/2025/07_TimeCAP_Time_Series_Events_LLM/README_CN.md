# TimeCAP: 基于大语言模型智能体的时序事件上下文增强与预测

## 论文信息

- **论文标题**: TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents
- **会议**: AAAI 2025
- **作者**: geon0325 等
- **GitHub**: https://github.com/geon0325/TimeCAP
- **论文**: https://arxiv.org/abs/2502.11418

## 核心内容

TimeCAP 是一种利用大语言模型智能体进行时序事件上下文增强与预测的创新框架。

### 主要贡献

1. **时序上下文化**: 使用 LLM 生成时序数据的文本描述，提供丰富的上下文信息
2. **三阶段预测流程**:
   - P1: 基于时序生成文本上下文 (Contextualization)
   - P2: 直接基于时序预测 (Prediction based on Time Series)
   - P3: 基于文本上下文预测 (Prediction based on Text)
3. **在多个领域验证有效性**

### 数据集

包含来自三个领域的7个时序数据集:

- 🌤️ 天气: weather_ny, weather_sf, weather_hs
- 💰 金融: finance_sp500, finance_nikkei  
- 🏥 医疗: healthcare_mortality, healthcare_positive

## 代码结构

```
07_TimeCAP_Time_Series_Events_LLM/
 dataset/          # 数据集目录
    weather_*/    # 天气数据
    finance_*/   # 金融数据
    healthcare_*/ # 医疗数据
 README.md         # 英文原版说明
```

## 使用方法

### 加载时序数据

```python
import pickle as pkl

with open('indices.pkl', 'rb') as f:
    indices = pkl.load(f)

with open(f'time_series_{city}.pkl', 'rb') as f:
    data = pkl.load(f)
```

### 加载标签

```python
with open(f'rain_{city}.pkl', 'rb') as f:
    labels = pkl.load(f)
```

### 三个预测阶段

1. **Contextualization**: 将时序数据转换为文本描述
2. **Time Series Prediction**: 直接使用时序数据进行预测
3. **Text-based Prediction**: 使用生成的文本描述进行预测

---

**来源**: 官方代码 (Official Code)
