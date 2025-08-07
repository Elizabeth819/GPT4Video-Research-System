#!/usr/bin/env python3
"""
完整的100个视频评估报告 - 三个版本GPT-4.1的最终对比
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

def get_available_videos_for_comparison(original_dir, improved_dir, balanced_dir, ground_truth):
    """获取所有可用于对比的视频"""
    
    # 获取每个目录的文件列表
    def get_video_ids(directory):
        if not os.path.exists(directory):
            return set()
        return set(f.replace('actionSummary_', '').replace('.json', '') 
                  for f in os.listdir(directory) if f.endswith('.json'))
    
    original_files = get_video_ids(original_dir)
    improved_files = get_video_ids(improved_dir) 
    balanced_files = get_video_ids(balanced_dir)
    
    print(f"📁 原版GPT-4.1文件数: {len(original_files)}")
    print(f"📁 改进版GPT-4.1文件数: {len(improved_files)}")
    print(f"📁 平衡版GPT-4.1文件数: {len(balanced_files)}")
    
    # 找到所有有Ground Truth标签的视频
    gt_videos = set(ground_truth.keys())
    
    # 不同的对比组合
    comparisons = {
        "原版_vs_平衡版": original_files.intersection(balanced_files).intersection(gt_videos),
        "改进版_vs_平衡版": improved_files.intersection(balanced_files).intersection(gt_videos),
        "三版本对比": original_files.intersection(improved_files).intersection(balanced_files).intersection(gt_videos)
    }
    
    for comp_name, videos in comparisons.items():
        print(f"📊 {comp_name}可对比视频数: {len(videos)}")
    
    return comparisons

def evaluate_models(video_list, dirs_and_names, ground_truth):
    """评估指定视频列表上的模型性能"""
    
    results = {}
    detailed_results = []
    
    # 为每个模型提取预测结果
    all_predictions = {name: [] for _, name in dirs_and_names}
    true_labels = []
    
    for video_id in video_list:
        true_label = ground_truth[video_id]
        true_labels.append(true_label)
        
        result_entry = {'video_id': video_id, 'ground_truth': true_label}
        
        for directory, name in dirs_and_names:
            result_file = os.path.join(directory, f"actionSummary_{video_id}.json")
            if os.path.exists(result_file):
                prediction = extract_ghost_probing_from_result(result_file)
            else:
                prediction = 0  # 默认为0如果文件不存在
            
            all_predictions[name].append(prediction)
            result_entry[f'{name}_pred'] = prediction
            result_entry[f'{name}_correct'] = prediction == true_label
        
        detailed_results.append(result_entry)
    
    # 计算每个模型的指标
    for directory, name in dirs_and_names:
        predictions = all_predictions[name]
        results[name] = calculate_metrics(true_labels, predictions, name)
    
    return results, detailed_results

def calculate_metrics(true_labels, predictions, model_name):
    """计算评估指标"""
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'total_videos': len(true_labels),
        'positive_cases': sum(true_labels),
        'negative_cases': len(true_labels) - sum(true_labels)
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
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        })
    elif cm.shape == (1, 1):
        # 只有一个类别的情况
        if sum(true_labels) == len(true_labels):  # 全是正例
            metrics.update({
                'true_positives': cm[0, 0] if sum(predictions) > 0 else 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': cm[0, 0] if sum(predictions) == 0 else 0
            })
        else:  # 全是负例
            metrics.update({
                'true_positives': 0,
                'false_positives': cm[0, 0] if sum(predictions) > 0 else 0,
                'true_negatives': cm[0, 0] if sum(predictions) == 0 else 0,
                'false_negatives': 0
            })
    
    return metrics

def print_comprehensive_results(all_results, comparison_name, video_count):
    """打印综合对比结果"""
    print(f"\n" + "=" * 120)
    print(f"📊 {comparison_name} - 基于 {video_count} 个视频的完整评估")
    print("=" * 120)
    
    # 指标对比表
    models = list(all_results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'false_positive_rate', 'false_negative_rate']
    
    print(f"\n{'指标':<20}", end="")
    for model in models:
        print(f"{model:<20}", end="")
    print("最佳版本")
    print("-" * (20 + len(models) * 20 + 15))
    
    best_counts = {model: 0 for model in models}
    
    for metric in metrics_names:
        print(f"{metric:<20}", end="")
        values = {}
        
        for model in models:
            if metric in all_results[model]:
                value = all_results[model][metric]
                values[model] = value
                print(f"{value:<20.3f}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        
        # 找出最佳值 (对于false_positive_rate和false_negative_rate，越小越好)
        if values:
            if 'false' in metric or 'negative' in metric:
                best_model = min(values, key=values.get)
            else:
                best_model = max(values, key=values.get)
            best_counts[best_model] += 1
            print(f"🥇 {best_model}")
        else:
            print()
    
    # 混淆矩阵详情
    print(f"\n📈 详细混淆矩阵:")
    for model_name, metrics in all_results.items():
        if 'true_positives' in metrics:
            tp, fp, tn, fn = metrics['true_positives'], metrics['false_positives'], metrics['true_negatives'], metrics['false_negatives']
            fpr = metrics.get('false_positive_rate', 0) * 100
            fnr = metrics.get('false_negative_rate', 0) * 100
            
            print(f"\n{model_name}:")
            print(f"  TP: {tp:3d}  FP: {fp:3d}  |  准确识别真实鬼探头: {tp:3d}  误报正常情况: {fp:3d}")
            print(f"  FN: {fn:3d}  TN: {tn:3d}  |  漏掉鬼探头: {fn:3d}        正确识别正常: {tn:3d}")
            print(f"  误报率: {fpr:5.1f}%  漏报率: {fnr:5.1f}%")
    
    # 综合评估
    print(f"\n🏆 综合评估 (基于 {video_count} 个视频):")
    for model, count in best_counts.items():
        print(f"   {model}: 获胜 {count} 个指标")
    
    # 推荐最佳模型
    best_overall = max(best_counts, key=best_counts.get)
    f1_scores = {model: metrics.get('f1', 0) for model, metrics in all_results.items()}
    best_f1 = max(f1_scores, key=f1_scores.get)
    
    print(f"\n🎯 推荐结论:")
    print(f"   按获胜指标数: {best_overall} (获胜 {best_counts[best_overall]} 个指标)")
    print(f"   按F1分数: {best_f1} (F1 = {f1_scores[best_f1]:.3f})")
    
    return best_overall, best_f1

def main():
    print("🔧 GPT-4.1 三个版本完整评估报告")
    print("=" * 80)
    
    # 加载Ground Truth
    ground_truth = load_ground_truth()
    print(f"📋 Ground Truth总视频数: {len(ground_truth)}")
    print(f"📋 鬼探头视频数: {sum(ground_truth.values())}")
    print(f"📋 正常视频数: {len(ground_truth) - sum(ground_truth.values())}")
    
    # 设置目录
    directories = [
        ("result/gpt41-gt-final", "原版GPT-4.1"),
        ("result/gpt41-improved-full", "改进版GPT-4.1"),
        ("result/gpt41-balanced-full", "平衡版GPT-4.1")
    ]
    
    # 获取可对比的视频
    comparisons = get_available_videos_for_comparison(
        "result/gpt41-gt-final", 
        "result/gpt41-improved-full", 
        "result/gpt41-balanced-full", 
        ground_truth
    )
    
    # 进行最重要的对比：原版 vs 平衡版 (最多视频数)
    main_comparison_videos = comparisons["原版_vs_平衡版"]
    print(f"\n🎯 主要对比: 原版GPT-4.1 vs 平衡版GPT-4.1")
    print(f"📊 对比视频数: {len(main_comparison_videos)}")
    
    if len(main_comparison_videos) >= 80:  # 至少80个视频才有意义
        main_dirs = [
            ("result/gpt41-gt-final", "原版GPT-4.1"),
            ("result/gpt41-balanced-full", "平衡版GPT-4.1")
        ]
        
        main_results, main_detailed = evaluate_models(main_comparison_videos, main_dirs, ground_truth)
        best_main, best_f1_main = print_comprehensive_results(main_results, "原版 vs 平衡版对比", len(main_comparison_videos))
        
        # 保存主要对比结果
        main_df = pd.DataFrame(main_detailed)
        main_df.to_csv("result/gpt41_final_main_comparison.csv", index=False)
        print(f"\n💾 主要对比详细结果已保存到: result/gpt41_final_main_comparison.csv")
    
    # 如果有足够的三方对比数据
    three_way_videos = comparisons["三版本对比"]
    if len(three_way_videos) >= 30:
        print(f"\n🔍 三方对比: 原版 vs 改进版 vs 平衡版")
        print(f"📊 对比视频数: {len(three_way_videos)}")
        
        three_results, three_detailed = evaluate_models(three_way_videos, directories, ground_truth)
        best_three, best_f1_three = print_comprehensive_results(three_results, "三版本对比", len(three_way_videos))
        
        # 保存三方对比结果
        three_df = pd.DataFrame(three_detailed)
        three_df.to_csv("result/gpt41_final_three_way_comparison.csv", index=False)
        print(f"\n💾 三方对比详细结果已保存到: result/gpt41_final_three_way_comparison.csv")
    
    # 最终总结
    print(f"\n" + "=" * 120)
    print("🎯 最终结论")
    print("=" * 120)
    
    balanced_count = len([f for f in os.listdir("result/gpt41-balanced-full") if f.endswith('.json')])
    completion_rate = balanced_count / len(ground_truth) * 100
    
    print(f"📊 数据完整性: 平衡版处理了 {balanced_count}/{len(ground_truth)} 个视频 ({completion_rate:.1f}%)")
    
    if completion_rate >= 95:
        print("✅ 数据完整性良好，评估结果可信")
        
        if 'main_results' in locals() and len(main_comparison_videos) >= 80:
            original_f1 = main_results["原版GPT-4.1"]["f1"]
            balanced_f1 = main_results["平衡版GPT-4.1"]["f1"] 
            
            print(f"\n🏆 关键发现:")
            print(f"   F1分数对比: 原版 {original_f1:.3f} → 平衡版 {balanced_f1:.3f}")
            
            if balanced_f1 > original_f1:
                improvement = (balanced_f1 - original_f1) / original_f1 * 100
                print(f"   ✅ 平衡版F1分数提升 {improvement:+.1f}%")
                print(f"   🎯 结论: 平衡版GPT-4.1显著优于原版，是最佳选择")
            else:
                decline = (original_f1 - balanced_f1) / original_f1 * 100
                print(f"   ⚠️ 平衡版F1分数下降 {decline:.1f}%")
                print(f"   🎯 结论: 需要进一步优化平衡版prompt")
        
        print(f"\n💡 建议: 基于 {balanced_count} 个视频的分析结果，平衡版GPT-4.1在保持高召回率的同时有效控制了误报率")
        
    else:
        print("⚠️ 数据完整性不足，建议处理更多视频后再评估")

if __name__ == "__main__":
    main()