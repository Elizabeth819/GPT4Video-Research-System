# Gemini-1.5-flash 100视频Ghost Probing检测最终报告

## 实验概述

- **实验时间**: 20250725_213943
- **模型**: Gemini-1.5-flash
- **Prompt版本**: balanced_gpt41_style
- **计划测试视频数**: 98
- **成功处理数**: 24
- **API配额限制失败数**: 74
- **实际成功率**: 24.5%

## API配额限制问题

- **第一个API密钥**: After ~24 videos
- **第二个API密钥**: Immediately (0 additional videos)
- **总配额错误数**: 74

## 基于成功处理的24个视频的性能指标

| 指标 | 数值 |
|------|------|
| 精确度 (Precision) | 0.600 |
| 召回率 (Recall) | 0.200 |
| F1分数 | 0.300 |
| 准确率 (Accuracy) | 0.417 |

## 详细统计

| 分类 | 数量 |
|------|------|
| True Positives (TP) | 3 |
| False Positives (FP) | 2 |
| True Negatives (TN) | 7 |
| False Negatives (FN) | 12 |
| API配额错误 (ERROR) | 74 |

## 实验限制

本次实验受到Gemini API配额限制的严重影响:

1. **第一个API密钥** (`AIzaSyDCWXFN2MaPaEab8B5dHSiSt9RkVww3AZ8`): 成功处理24个视频后达到配额限制
2. **第二个API密钥** (`AIzaSyA2nNsiLj7MJRSz99w3dtShozrNSBTdHCs`): 立即达到配额限制，无法处理任何额外视频
3. **总计**: 74个视频因为API配额限制无法处理

## 结论

基于成功处理的24个视频，Gemini-1.5-flash模型使用balanced版本的GPT-4.1风格prompt在ghost probing检测任务上的表现为:

- **F1分数**: 0.300
- **精确度**: 0.600 
- **召回率**: 0.200

**注意**: 由于API配额限制，这个结果仅基于24/98个视频，可能不能完全代表模型在全部数据集上的真实性能。需要额外的API配额来完成完整的100视频评估。

实验数据保存在: `/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gemini-1.5-flash-run1`

---
*报告生成时间: 20250725_213943*
*API配额限制导致实验不完整*
