# Gemini-1.5-flash 100视频Ghost Probing检测报告

## 实验概述

- **实验时间**: 20250725_211742
- **模型**: Gemini-1.5-flash
- **Prompt版本**: balanced_gpt41_style
- **测试视频总数**: 98
- **成功处理数**: 24
- **成功率**: 24.5%

## 性能指标

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
| 错误 (ERROR) | 74 |

## 结论

本次100视频测试使用了balanced版本的GPT-4.1风格prompt，在Gemini-1.5-flash模型上取得了F1分数0.300的性能表现。

实验数据保存在: `/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gemini-1.5-flash-run1`

---
*报告生成时间: 20250725_211742*
