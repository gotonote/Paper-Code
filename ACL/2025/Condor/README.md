# Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis

> ACL 2025

## 论文信息

- **标题**: Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement
- **作者**: InternLM
- **链接**: [GitHub](https://github.com/InternLM/Condor)

## 核心贡献

1. 提出知识驱动的LLM对齐数据合成方法
2. 提高模型对齐训练的质量和效率
3. 在多个对齐基准上取得提升

## 方法概述

Condor通过知识图谱引导的数据合成和优化，生成高质量的对齐训练数据。

## 运行方式

```bash
pip install -r requirements.txt
python synthesize.py --topic safety
```

## 依赖

- PyTorch >= 2.0
- transformers
- openai
