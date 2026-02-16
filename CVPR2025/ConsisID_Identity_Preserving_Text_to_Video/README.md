# ConsisID: Identity-Preserving Text-to-Video Generation

## 论文信息

- **论文标题**: Identity-Preserving Text-to-Video Generation by Frequency Decomposition
- **作者**: PKU-Yuan Group
- **机构**: Peking University
- **论文链接**: https://arxiv.org/abs/2411.17440
- **项目主页**: https://github.com/PKU-YuanGroup/ConsisID
- **代码仓库**: https://github.com/PKU-YuanGroup/ConsisID

## 核心贡献总结

1. **身份保持视频生成**: 实现保持人物身份的文本到视频生成，确保视频中人物的一致性。

2. **频率分解**: 提出频率分解方法，分别处理人物身份特征和动作/外观变化。

3. **高质量结果**: 生成的视频既保持人物身份特征，又具有流畅自然的动作。

## 方法概述

ConsisID的核心技术：

- **身份编码器**: 提取并编码人物身份特征
- **频率感知模块**: 将身份信息与动作信息分离处理
- **时序一致性**: 确保视频帧间的时间一致性

## 代码结构说明

```
ConsisID_Identity_Preserving_Text_to_Video/
├── models/                     # 模型定义
├── scripts/                    # 推理脚本
├── requirements.txt           # 依赖
└── ...
```

## 运行方式

### 环境安装
```bash
pip install -r requirements.txt
```

### 推理
```bash
# 参考项目README进行推理
```

## 依赖

- PyTorch
- diffusers
- transformers

## 许可证

研究用途
