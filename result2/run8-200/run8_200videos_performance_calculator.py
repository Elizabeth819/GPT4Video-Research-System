#!/usr/bin/env python3
"""
Run 8-200性能指标计算器
计算200视频数据集上的详细性能指标并与Run 8 (100视频)对比
"""

import json
import pandas as pd
from collections import Counter
import numpy as np
from datetime import datetime

def calculate_performance_metrics(results_file):
    """计算详细性能指标"""
    
    # 加载结果数据
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detailed_results = data['detailed_results']
    
    # 统计混淆矩阵
    tp = sum(1 for r in detailed_results if r['evaluation'] == 'TP')
    tn = sum(1 for r in detailed_results if r['evaluation'] == 'TN') 
    fp = sum(1 for r in detailed_results if r['evaluation'] == 'FP')
    fn = sum(1 for r in detailed_results if r['evaluation'] == 'FN')
    unknown = sum(1 for r in detailed_results if r['evaluation'] == 'UNKNOWN')
    
    total_processed = tp + tn + fp + fn + unknown
    valid_evaluations = tp + tn + fp + fn  # 排除unknown
    
    print("=" * 60)
    print("📊 Run 8-200视频实验性能分析报告")
    print("=" * 60)
    print(f"🎯 实验信息:")
    print(f"   - 实验ID: {data['experiment_info']['run_id']}")
    print(f"   - 时间戳: {data['experiment_info']['timestamp']}")
    print(f"   - 模型: {data['experiment_info']['model']}")
    print(f"   - Prompt版本: {data['experiment_info']['prompt_version']}")
    print(f"   - Temperature: {data['experiment_info']['temperature']}")
    print()
    
    print(f"📈 处理统计:")
    print(f"   - 目标视频数: {data['experiment_info']['video_count']}")
    print(f"   - 成功处理: {total_processed} 个视频")
    print(f"   - 有效评估: {valid_evaluations} 个视频") 
    print(f"   - 处理成功率: {total_processed/data['experiment_info']['video_count']*100:.1f}%")
    print()
    
    print(f"🎲 混淆矩阵统计:")
    print(f"   - True Positives (TP): {tp}")
    print(f"   - True Negatives (TN): {tn}")
    print(f"   - False Positives (FP): {fp}")
    print(f"   - False Negatives (FN): {fn}")
    print(f"   - Unknown/失败: {unknown}")
    print()
    
    if valid_evaluations > 0:
        # 计算性能指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / valid_evaluations
        
        # 平衡准确率
        sensitivity = recall  # 敏感性 = 召回率
        balanced_accuracy = (sensitivity + specificity) / 2
        
        print(f"📊 核心性能指标:")
        print(f"   - F1 Score: {f1_score:.3f} ({f1_score*100:.1f}%)")
        print(f"   - Precision: {precision:.3f} ({precision*100:.1f}%)")
        print(f"   - Recall: {recall:.3f} ({recall*100:.1f}%)")
        print(f"   - Specificity: {specificity:.3f} ({specificity*100:.1f}%)")
        print(f"   - Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   - Balanced Accuracy: {balanced_accuracy:.3f} ({balanced_accuracy*100:.1f}%)")
        print()
        
        # 数据集分布分析
        ghost_probing_count = tp + fn  # 实际ghost probing数量
        normal_count = tn + fp  # 实际normal数量
        
        print(f"📋 数据集分布:")
        print(f"   - Ghost Probing案例: {ghost_probing_count} ({ghost_probing_count/valid_evaluations*100:.1f}%)")
        print(f"   - Normal案例: {normal_count} ({normal_count/valid_evaluations*100:.1f}%)")
        print()
        
        # 预测分布
        predicted_positive = tp + fp
        predicted_negative = tn + fn
        
        print(f"🔮 预测分布:")
        print(f"   - 预测为Ghost Probing: {predicted_positive} ({predicted_positive/valid_evaluations*100:.1f}%)")
        print(f"   - 预测为Normal: {predicted_negative} ({predicted_negative/valid_evaluations*100:.1f}%)")
        print()
        
        # 错误分析
        print(f"❌ 错误分析:")
        print(f"   - 误报率 (FPR): {fp/(tn+fp)*100:.1f}% - {fp}个normal被误识别为ghost probing")
        print(f"   - 漏报率 (FNR): {fn/(tp+fn)*100:.1f}% - {fn}个ghost probing被误识别为normal")
        print()
        
        # 与Run 8 (100视频)对比数据
        print("=" * 60)
        print("📈 与Run 8 (100视频)性能对比")
        print("=" * 60)
        print("Run 8 (100视频) 参考指标:")
        print("   - F1 Score: 65.0%")
        print("   - Precision: 54.2%")
        print("   - Recall: 81.2%")
        print("   - Specificity: 67.4%") 
        print("   - Accuracy: 72.0%")
        print()
        
        # 性能变化分析
        f1_diff = f1_score * 100 - 65.0
        precision_diff = precision * 100 - 54.2
        recall_diff = recall * 100 - 81.2
        specificity_diff = specificity * 100 - 67.4
        accuracy_diff = accuracy * 100 - 72.0
        
        print(f"📊 Run 8-200 vs Run 8性能变化:")
        print(f"   - F1 Score: {f1_score*100:.1f}% ({f1_diff:+.1f})")
        print(f"   - Precision: {precision*100:.1f}% ({precision_diff:+.1f})")
        print(f"   - Recall: {recall*100:.1f}% ({recall_diff:+.1f})")
        print(f"   - Specificity: {specificity*100:.1f}% ({specificity_diff:+.1f})")
        print(f"   - Accuracy: {accuracy*100:.1f}% ({accuracy_diff:+.1f})")
        print()
        
        # 95%置信区间估计
        n = valid_evaluations
        f1_se = np.sqrt(f1_score * (1 - f1_score) / n)
        f1_ci_lower = max(0, f1_score - 1.96 * f1_se)
        f1_ci_upper = min(1, f1_score + 1.96 * f1_se)
        
        precision_se = np.sqrt(precision * (1 - precision) / (tp + fp)) if (tp + fp) > 0 else 0
        precision_ci_lower = max(0, precision - 1.96 * precision_se)
        precision_ci_upper = min(1, precision + 1.96 * precision_se)
        
        recall_se = np.sqrt(recall * (1 - recall) / (tp + fn)) if (tp + fn) > 0 else 0
        recall_ci_lower = max(0, recall - 1.96 * recall_se)
        recall_ci_upper = min(1, recall + 1.96 * recall_se)
        
        print(f"📐 95%置信区间:")
        print(f"   - F1 Score: [{f1_ci_lower*100:.1f}%, {f1_ci_upper*100:.1f}%]")
        print(f"   - Precision: [{precision_ci_lower*100:.1f}%, {precision_ci_upper*100:.1f}%]")
        print(f"   - Recall: [{recall_ci_lower*100:.1f}%, {recall_ci_upper*100:.1f}%]")
        print()
        
        # 数据规模效应分析
        print(f"📏 数据规模效应分析:")
        print(f"   - Run 8数据规模: 100 videos")
        print(f"   - Run 8-200数据规模: {valid_evaluations} videos")
        print(f"   - 规模增长: {valid_evaluations/100:.1f}x")
        print(f"   - 有效处理率: {valid_evaluations/200*100:.1f}%")
        
        # 结论和建议
        print("=" * 60)
        print("🎯 总结与建议")
        print("=" * 60)
        
        if f1_score >= 0.65:
            print("✅ 优秀表现: F1-score保持在65%以上，性能稳定")
        elif f1_score >= 0.60:
            print("⚠️  良好表现: F1-score在60-65%区间，有提升空间")
        else:
            print("❌ 需要改进: F1-score低于60%，建议优化")
            
        if precision >= 0.55:
            print("✅ 精确率良好: 误报控制在合理范围")
        else:
            print("⚠️  精确率偏低: 存在较多误报，建议优化判断逻辑")
            
        if recall >= 0.80:
            print("✅ 召回率优秀: 安全关键场景检测能力强")
        elif recall >= 0.70:
            print("⚠️  召回率中等: 部分ghost probing可能被遗漏")
        else:
            print("❌ 召回率偏低: 存在安全风险，建议提高检测敏感度")
        
        print()
        print("📝 关键发现:")
        print(f"   1. 在{valid_evaluations}个有效视频上保持了{'稳定' if abs(f1_diff) < 5 else '显著变化'}的性能")
        print(f"   2. 相比100视频数据集，F1-score{'提升' if f1_diff > 0 else '下降'}了{abs(f1_diff):.1f}%")
        print(f"   3. {'精确率' if precision_diff > recall_diff else '召回率'}相对表现更好")
        print(f"   4. 适合{'生产环境部署' if f1_score >= 0.60 and recall >= 0.75 else '进一步优化后部署'}")
        
        return {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1_score': f1_score,
            'specificity': specificity, 'accuracy': accuracy, 
            'balanced_accuracy': balanced_accuracy,
            'total_processed': total_processed, 'valid_evaluations': valid_evaluations
        }
    
    else:
        print("❌ 无有效评估数据，无法计算性能指标")
        return None

if __name__ == "__main__":
    results_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-200/run8_200videos_final_results_20250730_134411.json"
    metrics = calculate_performance_metrics(results_file)
    
    if metrics:
        # 保存性能指标到JSON文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-200/run8_200videos_performance_metrics_{timestamp}.json"
        
        metrics_data = {
            "experiment_info": {
                "experiment_id": "Run 8-200",
                "timestamp": timestamp,
                "video_dataset": "DADA-200-videos",
                "total_videos": 200,
                "processed_videos": metrics['total_processed'],
                "valid_evaluations": metrics['valid_evaluations']
            },
            "performance_metrics": {
                "f1_score": metrics['f1_score'],
                "precision": metrics['precision'], 
                "recall": metrics['recall'],
                "specificity": metrics['specificity'],
                "accuracy": metrics['accuracy'],
                "balanced_accuracy": metrics['balanced_accuracy']
            },
            "confusion_matrix": {
                "true_positives": metrics['tp'],
                "true_negatives": metrics['tn'],
                "false_positives": metrics['fp'],
                "false_negatives": metrics['fn']
            },
            "comparison_with_run8": {
                "run8_f1": 0.65,
                "run8_precision": 0.542,
                "run8_recall": 0.812,
                "run8_specificity": 0.674,
                "run8_accuracy": 0.72,
                "f1_improvement": metrics['f1_score'] - 0.65,
                "precision_improvement": metrics['precision'] - 0.542,
                "recall_improvement": metrics['recall'] - 0.812,
                "specificity_improvement": metrics['specificity'] - 0.674,
                "accuracy_improvement": metrics['accuracy'] - 0.72
            }
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 性能指标已保存到: {metrics_file}")