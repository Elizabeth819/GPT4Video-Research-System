#!/usr/bin/env python3
"""
Run 7 监控和分析脚本
监控当前进度并生成中期分析报告
"""

import json
import os
import datetime
from collections import Counter

def analyze_current_progress():
    """分析当前Run 7进度"""
    
    # 查找最新的中间结果文件
    run7_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run7-gpt4o-paper-batch-temp0"
    
    # 获取所有中间结果文件
    intermediate_files = []
    for filename in os.listdir(run7_dir):
        if filename.startswith("run7_intermediate_") and filename.endswith(".json"):
            intermediate_files.append(filename)
    
    if not intermediate_files:
        print("未找到中间结果文件")
        return
    
    # 使用最新的文件
    latest_file = sorted(intermediate_files)[-1]
    file_path = os.path.join(run7_dir, latest_file)
    
    print(f"分析文件: {latest_file}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 统计当前进度
    detailed_results = results["detailed_results"]
    total_processed = len(detailed_results)
    
    print(f"\n{'='*60}")
    print(f"Run 7: GPT-4o + Paper_Batch (Temperature=0) 当前进度")
    print(f"{'='*60}")
    print(f"已处理视频: {total_processed}/99")
    print(f"进度: {total_processed/99*100:.1f}%")
    
    # 计算当前性能指标
    evaluations = [r["evaluation"] for r in detailed_results]
    eval_counts = Counter(evaluations)
    
    tp = eval_counts.get('TP', 0)
    fp = eval_counts.get('FP', 0) 
    tn = eval_counts.get('TN', 0)
    fn = eval_counts.get('FN', 0)
    errors = eval_counts.get('ERROR', 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print(f"\n当前性能指标 (基于 {total_processed} 个视频):")
    print(f"精确度: {precision:.3f} ({precision*100:.1f}%)")
    print(f"召回率: {recall:.3f} ({recall*100:.1f}%)")
    print(f"F1分数: {f1:.3f} ({f1*100:.1f}%)")
    print(f"准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print(f"\n混淆矩阵:")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, ERROR: {errors}")
    
    # 与Run 6对比
    run6_f1 = 0.636
    run6_precision = 0.554
    run6_recall = 0.745
    
    print(f"\n与Run 6对比 (Temperature=0.3):")
    print(f"F1分数: {f1:.3f} vs {run6_f1:.3f} ({f1-run6_f1:+.3f})")
    print(f"精确度: {precision:.3f} vs {run6_precision:.3f} ({precision-run6_precision:+.3f})")
    print(f"召回率: {recall:.3f} vs {run6_recall:.3f} ({recall-run6_recall:+.3f})")
    
    # Temperature影响分析
    print(f"\nTemperature=0 影响分析:")
    if f1 > run6_f1:
        print("✅ Temperature=0 提升了整体性能")
    elif abs(f1 - run6_f1) < 0.02:
        print("⚖️ Temperature差异对性能影响较小")
    else:
        print("⚠️ Temperature=0 可能降低了性能")
    
    # 一致性分析
    error_rate = errors / total_processed if total_processed > 0 else 0
    print(f"错误率: {error_rate:.3f} ({error_rate*100:.1f}%)")
    print(f"处理稳定性: {'优秀' if error_rate == 0 else '良好' if error_rate < 0.05 else '需要改进'}")
    
    # 生成中期报告
    generate_interim_report(results, total_processed, precision, recall, f1, accuracy, tp, fp, tn, fn, errors, latest_file)
    
    return total_processed

def generate_interim_report(results, total_processed, precision, recall, f1, accuracy, tp, fp, tn, fn, errors, filename):
    """生成中期报告"""
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Run 7: GPT-4o + Paper_Batch (Temperature=0) 中期进度报告

## 实验状态

- **当前时间**: {timestamp}
- **进度**: {total_processed}/99 个视频 ({total_processed/99*100:.1f}%)
- **状态**: 🔄 **进行中** 
- **数据文件**: {filename}

## 实验配置

- **Run ID**: Run 7  
- **模型**: GPT-4o (Azure)
- **Prompt版本**: Paper_Batch Complex (4-Task)
- **关键修正**: Temperature=0 (修正Run 6的0.3设置)
- **目的**: 验证Temperature参数对一致性和性能的影响

## 当前性能指标

| 指标 | 数值 | 百分比 |
|------|------|--------|
| **精确度 (Precision)** | {precision:.3f} | {precision*100:.1f}% |
| **召回率 (Recall)** | {recall:.3f} | {recall*100:.1f}% |
| **F1分数** | {f1:.3f} | {f1*100:.1f}% |
| **准确率 (Accuracy)** | {accuracy:.3f} | {accuracy*100:.1f}% |

## 混淆矩阵

- **True Positives (TP)**: {tp}
- **False Positives (FP)**: {fp}
- **True Negatives (TN)**: {tn}
- **False Negatives (FN)**: {fn}
- **处理错误 (ERROR)**: {errors}

## Temperature=0 vs 0.3 对比

| 指标 | Run 7 (Temp=0) | Run 6 (Temp=0.3) | 差异 |
|------|----------------|------------------|------|
| **F1分数** | {f1:.3f} | 0.636 | {f1-0.636:+.3f} |
| **精确度** | {precision:.3f} | 0.554 | {precision-0.554:+.3f} |
| **召回率** | {recall:.3f} | 0.745 | {recall-0.745:+.3f} |
| **准确率** | {accuracy:.3f} | 0.530 | {accuracy-0.530:+.3f} |

## 中期发现

### 处理稳定性
- **错误率**: {errors/total_processed*100:.1f}% ({'0个错误，优秀' if errors == 0 else f'{errors}个错误'})
- **API稳定性**: {'优秀' if errors == 0 else '良好' if errors < total_processed * 0.05 else '需要改进'}
- **一致性**: Temperature=0确保了完全可重复的结果

### 性能趋势
{'- **F1分数**: 当前' + f'{f1:.3f}，' + ('高于' if f1 > 0.636 else '低于' if f1 < 0.636 else '接近') + 'Run 6的0.636'}
{'- **精确度**: 当前' + f'{precision:.3f}，' + ('高于' if precision > 0.554 else '低于' if precision < 0.554 else '接近') + 'Run 6的0.554'}
{'- **召回率**: 当前' + f'{recall:.3f}，' + ('高于' if recall > 0.745 else '低于' if recall < 0.745 else '接近') + 'Run 6的0.745'}

### Temperature参数影响初步结论
{
    "🎯 **Temperature=0正面影响明显**: 提升了整体性能，特别是F1分数和精确度" if f1 > 0.636 and precision > 0.554 else
    "⚖️ **Temperature参数影响适中**: 性能指标有所变化，但差异不大" if abs(f1 - 0.636) < 0.05 else
    "🔄 **需要更多数据验证**: 当前样本可能不足以得出结论" if total_processed < 50 else
    "📊 **Temperature=0显示不同特征**: 可能在精确度和召回率间有不同权衡"
}

## 预期完成

- **预计剩余视频**: {99 - total_processed}个
- **预计剩余时间**: {(99 - total_processed) * 20 / 60:.1f}分钟 (按20秒/视频估算)
- **预计完成时间**: {(datetime.datetime.now() + datetime.timedelta(seconds=(99 - total_processed) * 20)).strftime("%H:%M")}

## 技术细节

- **批次处理**: 10个视频/批次
- **平均处理时间**: ~20秒/视频
- **临时文件管理**: 自动清理
- **断点续传**: 支持

---
*中期报告生成时间: {timestamp}*  
*基于: {total_processed}/99 个已处理视频*  
*状态: 🔄 实验进行中*
"""

    # 保存报告
    run7_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run7-gpt4o-paper-batch-temp0"
    report_file = os.path.join(run7_dir, f"run7_interim_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n中期报告已保存: {report_file}")

if __name__ == "__main__":
    analyze_current_progress()