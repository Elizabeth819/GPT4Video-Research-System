# 消融实验报告: 5 Few-shot Samples

## 实验信息
- **实验时间**: 20250731_134159
- **实验目的**: 测试增强few-shot学习的效果 (边际效应分析)
- **基线对比**: Run 8 (3 few-shot samples, F1=70.0%)
- **处理视频数**: 92个

## 实验配置
- **模型**: GPT-4o (Azure)
- **Temperature**: 0
- **Prompt**: Paper_Batch Complex (4-Task)
- **Few-shot样本数**: 5个
  - Example 1: Ghost Probing Detection (pedestrian)
  - Example 2: Normal Driving (baseline)
  - Example 3: Vehicle Ghost Probing (complex)
  - Example 4: Cyclist Ghost Probing (additional)
  - Example 5: Highway Normal Driving (additional baseline)

## 性能结果

### 混淆矩阵
- True Positives (TP): 38
- True Negatives (TN): 11
- False Positives (FP): 33
- False Negatives (FN): 10

### 性能指标
- **F1 Score**: 0.639 (63.9%)
- **Precision**: 0.535 (53.5%)
- **Recall**: 0.792 (79.2%)
- **Specificity**: 0.250 (25.0%)
- **Accuracy**: 0.533 (53.3%)
- **Balanced Accuracy**: 0.521 (52.1%)

## 与基线对比 (Run 8: 3 samples)
- **F1差异**: 63.9% vs 70.0% = -6.1%
- **Recall差异**: 79.2% vs 84.8% = -5.6%
- **Precision差异**: 53.5% vs 59.6% = -6.1%

## 实验结论
1. **边际效应分析**: 5个样本相比3个样本的性能边际收益
2. **样本多样性**: 增加多样化样本是否能进一步提升性能
3. **计算成本权衡**: 更多样本带来的性能提升是否值得额外的计算成本

## 文件路径
- 详细结果: `ablation_5samples_results_20250731_134159.json`
- 实验日志: `ablation_5samples_20250731_134159.log`
