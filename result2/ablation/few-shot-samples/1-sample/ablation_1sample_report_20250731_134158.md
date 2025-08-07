# 消融实验报告: 1 Few-shot Sample

## 实验信息
- **实验时间**: 20250731_134158
- **实验目的**: 测试单个few-shot样本的学习效果
- **基线对比**: Run 8 (3 few-shot samples, F1=70.0%)
- **处理视频数**: 91个

## 实验配置
- **模型**: GPT-4o (Azure)
- **Temperature**: 0
- **Prompt**: Paper_Batch Complex (4-Task)
- **Few-shot样本数**: 1个 (Ghost Probing Detection样本)

## 性能结果

### 混淆矩阵
- True Positives (TP): 33
- True Negatives (TN): 15
- False Positives (FP): 31
- False Negatives (FN): 12

### 性能指标
- **F1 Score**: 0.606 (60.6%)
- **Precision**: 0.516 (51.6%)
- **Recall**: 0.733 (73.3%)
- **Specificity**: 0.326 (32.6%)
- **Accuracy**: 0.527 (52.7%)
- **Balanced Accuracy**: 0.530 (53.0%)

## 与基线对比 (Run 8: 3 samples)
- **F1差异**: 60.6% vs 70.0% = -9.4%
- **Recall差异**: 73.3% vs 84.8% = -11.5%
- **Precision差异**: 51.6% vs 59.6% = -8.0%

## 实验结论
1. **Few-shot学习效果**: 单个样本相比3个样本的性能变化
2. **最小学习能力**: 验证了最小few-shot学习的可行性
3. **性能权衡**: 分析了样本数量与性能的关系

## 文件路径
- 详细结果: `ablation_1sample_results_20250731_134158.json`
- 实验日志: `ablation_1sample_20250731_134158.log`
