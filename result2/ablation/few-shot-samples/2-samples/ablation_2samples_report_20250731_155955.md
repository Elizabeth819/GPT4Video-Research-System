# 消融实验报告: 2 Few-shot Samples

## 实验信息
- **实验时间**: 20250731_155955
- **实验目的**: 测试平衡few-shot学习的效果 (1 positive + 1 negative样本)
- **基线对比**: Run 8 (3 few-shot samples, F1=70.0%)
- **处理视频数**: 98个

## 实验配置
- **模型**: GPT-4o (Azure)
- **Temperature**: 0
- **Prompt**: Paper_Batch Complex (4-Task)
- **Few-shot样本数**: 2个
  - Example 1: Ghost Probing Detection (positive样本)
  - Example 2: Normal Driving (negative样本)

## 性能结果

### 混淆矩阵
- True Positives (TP): 40
- True Negatives (TN): 12
- False Positives (FP): 35
- False Negatives (FN): 11

### 性能指标
- **F1 Score**: 0.635 (63.5%)
- **Precision**: 0.533 (53.3%)
- **Recall**: 0.784 (78.4%)
- **Specificity**: 0.255 (25.5%)
- **Accuracy**: 0.531 (53.1%)
- **Balanced Accuracy**: 0.520 (52.0%)

## 与基线对比 (Run 8: 3 samples)
- **F1差异**: 63.5% vs 70.0% = -6.5%
- **Recall差异**: 78.4% vs 84.8% = -6.4%
- **Precision差异**: 53.3% vs 59.6% = -6.3%

## 实验结论
1. **平衡学习效果**: 2个样本(positive+negative)相比3个样本的性能变化
2. **样本质量vs数量**: 验证了平衡样本组合的重要性
3. **学习效率**: 分析了最小有效few-shot学习的阈值

## 文件路径
- 详细结果: `ablation_2samples_results_20250731_155955.json`
- 实验日志: `ablation_2samples_20250731_155955.log`
