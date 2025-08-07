#!/usr/bin/env python3
"""
评估平衡版GPT-4.1的进展 - 对比三个版本的性能
"""

import os
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

def load_ground_truth():
    """加载Ground Truth标签"""
    labels_file = "result/groundtruth_labels.csv"
    df = pd.read_csv(labels_file, sep='\t')
    
    # 解析标签
    ground_truth = {}
    for _, row in df.iterrows():
        video_id = row['video_id'].replace('.avi', '')
        label = row['ground_truth_label']
        
        # 转换为二进制标签
        has_ghost_probing = 0 if label == 'none' else 1
        ground_truth[video_id] = has_ghost_probing
    
    return ground_truth

def extract_ghost_probing_from_result(result_file):
    """从结果文件中提取是否包含ghost probing"""
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        # 检查所有段落中是否有ghost probing或potential ghost probing
        for segment in segments:
            if isinstance(segment, dict):
                key_actions = segment.get('key_actions', '').lower()
                if 'ghost probing' in key_actions:  # 包括 "ghost probing" 和 "potential ghost probing"
                    return 1
        return 0
    except Exception as e:
        print(f"❌ 解析文件失败: {result_file}, 错误: {str(e)}")
        return 0

def evaluate_three_models(original_dir, improved_dir, balanced_dir, ground_truth):
    """对比三个版本模型的性能"""
    
    # 找到三个目录都有的视频
    original_files = set(f.replace('actionSummary_', '').replace('.json', '') 
                        for f in os.listdir(original_dir) if f.endswith('.json'))
    improved_files = set(f.replace('actionSummary_', '').replace('.json', '') 
                        for f in os.listdir(improved_dir) if f.endswith('.json'))
    balanced_files = set(f.replace('actionSummary_', '').replace('.json', '') 
                        for f in os.listdir(balanced_dir) if f.endswith('.json'))
    
    # 找到所有三个版本都处理过的视频
    common_videos = original_files.intersection(improved_files).intersection(balanced_files)
    
    # 只评估有Ground Truth标签的视频
    valid_videos = [vid for vid in common_videos if vid in ground_truth]
    
    print(f"📊 三个版本都有的可评估视频数量: {len(valid_videos)}")
    
    if len(valid_videos) < 10:
        print("⚠️  可评估视频数量太少，结果可能不具有代表性")
    
    # 提取预测结果
    original_predictions = []
    improved_predictions = []
    balanced_predictions = []
    true_labels = []
    
    detailed_results = []
    
    for video_id in valid_videos:
        original_file = os.path.join(original_dir, f"actionSummary_{video_id}.json")
        improved_file = os.path.join(improved_dir, f"actionSummary_{video_id}.json")
        balanced_file = os.path.join(balanced_dir, f"actionSummary_{video_id}.json")
        
        original_pred = extract_ghost_probing_from_result(original_file)
        improved_pred = extract_ghost_probing_from_result(improved_file)
        balanced_pred = extract_ghost_probing_from_result(balanced_file)
        true_label = ground_truth[video_id]
        
        original_predictions.append(original_pred)
        improved_predictions.append(improved_pred)
        balanced_predictions.append(balanced_pred)
        true_labels.append(true_label)
        
        # 记录详细结果
        detailed_results.append({
            'video_id': video_id,
            'ground_truth': true_label,
            'original_pred': original_pred,
            'improved_pred': improved_pred,
            'balanced_pred': balanced_pred,
            'original_correct': original_pred == true_label,
            'improved_correct': improved_pred == true_label,
            'balanced_correct': balanced_pred == true_label
        })
    
    # 计算指标
    original_metrics = calculate_metrics(true_labels, original_predictions, "原版GPT-4.1")
    improved_metrics = calculate_metrics(true_labels, improved_predictions, "改进版GPT-4.1")
    balanced_metrics = calculate_metrics(true_labels, balanced_predictions, "平衡版GPT-4.1")
    
    return original_metrics, improved_metrics, balanced_metrics, detailed_results

def calculate_metrics(true_labels, predictions, model_name):
    """计算评估指标"""
    if len(set(true_labels)) == 1:
        print(f"⚠️  {model_name}: 所有真实标签都相同，某些指标可能不准确")
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0)
    }
    
    # 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        })
    
    return metrics

def print_three_way_comparison(original_metrics, improved_metrics, balanced_metrics, detailed_results):
    """打印三个版本的对比结果"""
    print("\n" + "=" * 100)
    print("📊 GPT-4.1 三个版本性能对比")
    print("=" * 100)
    
    # 指标对比表
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    
    print(f"\n{'指标':<15} {'原版GPT-4.1':<15} {'改进版GPT-4.1':<15} {'平衡版GPT-4.1':<15} {'最佳版本':<15}")
    print("-" * 85)
    
    best_counts = {'original': 0, 'improved': 0, 'balanced': 0}
    
    for metric in metrics_names:
        if metric in original_metrics and metric in improved_metrics and metric in balanced_metrics:
            original_val = original_metrics[metric]
            improved_val = improved_metrics[metric]
            balanced_val = balanced_metrics[metric]
            
            # 找出最佳值
            values = {'original': original_val, 'improved': improved_val, 'balanced': balanced_val}
            best_key = max(values, key=values.get)
            best_counts[best_key] += 1
            
            best_marker = {
                'original': '🥇' if best_key == 'original' else '',
                'improved': '🥇' if best_key == 'improved' else '', 
                'balanced': '🥇' if best_key == 'balanced' else ''
            }
            
            print(f"{metric:<15} {original_val:<15.3f} {improved_val:<15.3f} {balanced_val:<15.3f} {best_marker[best_key]:<15}")
    
    # 混淆矩阵对比
    if 'true_positives' in original_metrics:
        print(f"\n📈 混淆矩阵对比:")
        
        models = [
            ("原版GPT-4.1", original_metrics),
            ("改进版GPT-4.1", improved_metrics), 
            ("平衡版GPT-4.1", balanced_metrics)
        ]
        
        for model_name, metrics in models:
            print(f"\n{model_name}:")
            print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
            print(f"  FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")
            print(f"  误报率: {metrics['false_positives']/(metrics['false_positives']+metrics['true_negatives'])*100:.1f}%")
            print(f"  漏报率: {metrics['false_negatives']/(metrics['false_negatives']+metrics['true_positives'])*100:.1f}%")
    
    # 综合评估
    print(f"\n🏆 综合评估 (基于 {len(detailed_results)} 个视频):")
    print(f"   原版GPT-4.1 获胜指标: {best_counts['original']} 个")
    print(f"   改进版GPT-4.1 获胜指标: {best_counts['improved']} 个") 
    print(f"   平衡版GPT-4.1 获胜指标: {best_counts['balanced']} 个")
    
    # 关键改进点分析
    improved_cases = []
    for result in detailed_results:
        if result['balanced_correct'] and not result['original_correct']:
            improved_cases.append(f"{result['video_id']}: 平衡版修正了原版的错误")
        elif result['balanced_correct'] and not result['improved_correct']:
            improved_cases.append(f"{result['video_id']}: 平衡版修正了改进版的错误")
    
    if improved_cases:
        print(f"\n✅ 平衡版的关键改进 ({len(improved_cases)} 个案例):")
        for case in improved_cases[:5]:
            print(f"   • {case}")

def main():
    print("🔧 评估平衡版GPT-4.1的三方对比")
    print("=" * 60)
    
    # 加载Ground Truth
    ground_truth = load_ground_truth()
    print(f"📋 Ground Truth标签数量: {len(ground_truth)}")
    
    # 设置目录
    original_dir = "result/gpt41-gt-final"
    improved_dir = "result/gpt41-improved-full"
    balanced_dir = "result/gpt41-balanced-full"
    
    # 检查目录是否存在
    directories = [
        (original_dir, "原版"),
        (improved_dir, "改进版"),
        (balanced_dir, "平衡版")
    ]
    
    for dir_path, dir_name in directories:
        if not os.path.exists(dir_path):
            print(f"❌ {dir_name}结果目录不存在: {dir_path}")
            return
        
        count = len([f for f in os.listdir(dir_path) if f.endswith('.json')])
        print(f"📁 {dir_name}结果数量: {count}")
    
    # 进行三方评估
    original_metrics, improved_metrics, balanced_metrics, detailed_results = evaluate_three_models(
        original_dir, improved_dir, balanced_dir, ground_truth
    )
    
    # 打印结果
    print_three_way_comparison(original_metrics, improved_metrics, balanced_metrics, detailed_results)
    
    # 保存详细结果
    results_df = pd.DataFrame(detailed_results)
    results_file = "result/gpt41_three_way_comparison.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 详细结果已保存到: {results_file}")
    
    # 如果平衡版数量不够，提示继续处理
    balanced_count = len([f for f in os.listdir(balanced_dir) if f.endswith('.json')])
    if balanced_count < len(ground_truth) * 0.9:
        print(f"\n⚠️  平衡版还需要处理更多视频 (当前: {balanced_count}/{len(ground_truth)})")
        print("💡 建议继续运行批处理脚本以获得更完整的评估")
    else:
        print(f"\n🎉 平衡版处理完成度: {balanced_count}/{len(ground_truth)} ({balanced_count/len(ground_truth)*100:.1f}%)")

if __name__ == "__main__":
    main()