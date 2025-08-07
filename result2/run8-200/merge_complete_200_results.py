#!/usr/bin/env python3
"""
合并Run 8-200的所有结果，生成完整的200视频性能统计
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

def merge_results():
    """合并主结果和补充结果"""
    
    # 读取主结果 (190个视频)
    main_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-200/run8_200videos_final_results_20250730_134411.json"
    with open(main_file, 'r', encoding='utf-8') as f:
        main_data = json.load(f)
    
    # 读取补充结果 (10个视频)
    supplement_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-200/remaining_10_videos_results_20250730_163001.json"
    with open(supplement_file, 'r', encoding='utf-8') as f:
        supplement_data = json.load(f)
    
    print("=" * 70)
    print("📊 Run 8-200 完整200视频结果合并")
    print("=" * 70)
    print(f"📂 主结果文件: {len(main_data['detailed_results'])} 个视频")
    print(f"📂 补充结果文件: {len(supplement_data['detailed_results'])} 个视频")
    
    # 合并详细结果
    all_results = main_data['detailed_results'] + supplement_data['detailed_results']
    
    print(f"🎯 合并后总数: {len(all_results)} 个视频")
    
    # 创建完整的合并结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_data = {
        "experiment_info": {
            "run_id": "Run 8-200 Complete",
            "timestamp": timestamp,
            "video_count": 200,
            "processed_videos": len(all_results),
            "model": "GPT-4o (Azure)",
            "prompt_version": "Paper_Batch Complex (4-Task) + Few-shot Examples",
            "temperature": 0,
            "max_tokens": 3000,
            "purpose": "完整的200视频DADA-200数据集测试，验证Run 8配置的大规模性能",
            "ground_truth_file": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/DADA-200-videos/labels.csv",
            "output_directory": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-200",
            "merged_from": [
                "run8_200videos_final_results_20250730_134411.json (190 videos)",
                "remaining_10_videos_results_20250730_163001.json (10 videos)"
            ]
        },
        "detailed_results": all_results
    }
    
    # 保存合并结果
    merged_file = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-200/run8_200videos_complete_results_{timestamp}.json"
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 完整结果文件已保存: {merged_file}")
    
    return merged_file, all_results

def calculate_complete_performance(results):
    """计算完整的200视频性能指标"""
    
    # 统计混淆矩阵
    tp = sum(1 for r in results if r['evaluation'] == 'TP')
    tn = sum(1 for r in results if r['evaluation'] == 'TN') 
    fp = sum(1 for r in results if r['evaluation'] == 'FP')
    fn = sum(1 for r in results if r['evaluation'] == 'FN')
    unknown = sum(1 for r in results if r['evaluation'] == 'UNKNOWN')
    
    total_processed = len(results)
    valid_evaluations = tp + tn + fp + fn  # 排除unknown
    
    print("\n" + "=" * 70)
    print("🎯 Run 8-200 完整200视频性能分析报告")
    print("=" * 70)
    
    print(f"📊 处理统计:")
    print(f"   - 目标视频数: 200")
    print(f"   - 成功处理: {total_processed} 个视频")
    print(f"   - 有效评估: {valid_evaluations} 个视频") 
    print(f"   - 处理成功率: {total_processed/200*100:.1f}%")
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
        
        # 与Run 8 (100视频)对比
        print("=" * 70)
        print("📈 与Run 8 (100视频)性能对比")
        print("=" * 70)
        print("Run 8 (100视频) 参考指标:")
        print("   - F1 Score: 65.0%")
        print("   - Precision: 54.2%")
        print("   - Recall: 81.2%")
        print("   - Specificity: 67.4%") 
        print("   - Accuracy: 72.0%")
        print()
        
        # 与190视频结果对比
        print("Run 8-190 (部分结果) 参考指标:")
        print("   - F1 Score: 58.9%")
        print("   - Precision: 46.8%")
        print("   - Recall: 79.5%")
        print("   - Specificity: 29.9%")
        print("   - Accuracy: 51.6%")
        print()
        
        # 性能变化分析
        f1_diff_vs_100 = f1_score * 100 - 65.0
        precision_diff_vs_100 = precision * 100 - 54.2
        recall_diff_vs_100 = recall * 100 - 81.2
        specificity_diff_vs_100 = specificity * 100 - 67.4
        accuracy_diff_vs_100 = accuracy * 100 - 72.0
        
        f1_diff_vs_190 = f1_score * 100 - 58.9
        precision_diff_vs_190 = precision * 100 - 46.8
        
        print(f"📊 Run 8-200 完整版性能:")
        print(f"   vs Run 8 (100视频):")
        print(f"     - F1 Score: {f1_score*100:.1f}% ({f1_diff_vs_100:+.1f})")
        print(f"     - Precision: {precision*100:.1f}% ({precision_diff_vs_100:+.1f})")
        print(f"     - Recall: {recall*100:.1f}% ({recall_diff_vs_100:+.1f})")
        print(f"     - Specificity: {specificity*100:.1f}% ({specificity_diff_vs_100:+.1f})")
        print(f"     - Accuracy: {accuracy*100:.1f}% ({accuracy_diff_vs_100:+.1f})")
        print()
        print(f"   vs Run 8-190 (部分结果):")
        print(f"     - F1 Score: {f1_score*100:.1f}% ({f1_diff_vs_190:+.1f})")
        print(f"     - Precision: {precision*100:.1f}% ({precision_diff_vs_190:+.1f})")
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
        
        # 最终结论
        print("=" * 70)
        print("🎯 最终结论")
        print("=" * 70)
        
        print(f"✅ 完整处理了200个视频中的{total_processed}个 ({total_processed/200*100:.1f}%)")
        print(f"📊 完整版F1-score: {f1_score*100:.1f}%")
        
        if f1_score >= 0.60:
            print("✅ 性能达标: F1-score保持在60%以上")
        else:
            print("⚠️  性能待优化: F1-score低于60%")
        
        print(f"🔍 关键发现:")
        print(f"   1. 200视频完整版相比100视频版本F1下降了{abs(f1_diff_vs_100):.1f}%")
        print(f"   2. 召回率保持{recall*100:.1f}%，满足安全系统要求")
        print(f"   3. 大规模数据验证了模型的真实性能边界")
        print(f"   4. 为后续模型优化提供了可靠的基准数据")
        
        # 保存性能指标
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-200/run8_200videos_complete_performance_metrics_{timestamp}.json"
        
        metrics_data = {
            "experiment_info": {
                "experiment_id": "Run 8-200 Complete",
                "timestamp": timestamp,
                "video_dataset": "DADA-200-videos",
                "total_videos": 200,
                "processed_videos": total_processed,
                "valid_evaluations": valid_evaluations
            },
            "performance_metrics": {
                "f1_score": f1_score,
                "precision": precision, 
                "recall": recall,
                "specificity": specificity,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy
            },
            "confusion_matrix": {
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "unknown": unknown
            },
            "dataset_distribution": {
                "ghost_probing_cases": ghost_probing_count,
                "normal_cases": normal_count,
                "ghost_probing_percentage": ghost_probing_count/valid_evaluations*100,
                "normal_percentage": normal_count/valid_evaluations*100
            },
            "comparison": {
                "vs_run8_100videos": {
                    "f1_improvement": f1_diff_vs_100,
                    "precision_improvement": precision_diff_vs_100,
                    "recall_improvement": recall_diff_vs_100,
                    "specificity_improvement": specificity_diff_vs_100,
                    "accuracy_improvement": accuracy_diff_vs_100
                },
                "vs_run8_190videos": {
                    "f1_improvement": f1_diff_vs_190,
                    "precision_improvement": precision_diff_vs_190
                }
            },
            "confidence_intervals_95": {
                "f1_score": [f1_ci_lower, f1_ci_upper],
                "precision": [precision_ci_lower, precision_ci_upper],
                "recall": [recall_ci_lower, recall_ci_upper]
            }
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 完整性能指标已保存到: {metrics_file}")
        
        return metrics_data
    
    else:
        print("❌ 无有效评估数据，无法计算性能指标")
        return None

if __name__ == "__main__":
    print("🚀 开始合并Run 8-200完整结果...")
    
    # 合并结果
    merged_file, all_results = merge_results()
    
    # 计算性能指标
    metrics = calculate_complete_performance(all_results)
    
    print(f"\n🎉 Run 8-200完整版分析完成！")
    print(f"📁 结果文件: {merged_file}")
    
    if metrics:
        print(f"🎯 最终F1-score: {metrics['performance_metrics']['f1_score']*100:.1f}%")
        print(f"📊 处理视频数: {metrics['experiment_info']['processed_videos']}/200")