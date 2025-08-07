#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于97个视频的Gemini vs GPT-4.1完整对比分析
"""

import os
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_gemini_results():
    """加载所有Gemini结果"""
    gemini_dir = "result/gemini-balanced-full"
    results = {}
    
    print("📊 加载Gemini结果...")
    for filename in os.listdir(gemini_dir):
        if filename.startswith("actionSummary_") and filename.endswith(".json"):
            video_id = filename.replace("actionSummary_", "").replace(".json", "")
            # 标准化为images_格式
            if video_id.startswith("dada_"):
                video_id = video_id.replace("dada_", "images_")
            
            with open(os.path.join(gemini_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取key_actions
            key_actions = []
            for segment in data:
                if isinstance(segment, dict) and 'key_actions' in segment:
                    key_actions.append(segment['key_actions'])
            
            results[video_id] = key_actions
    
    print(f"✅ 加载了 {len(results)} 个Gemini视频结果")
    return results

def load_gpt41_results():
    """加载GPT-4.1平衡版结果"""
    gpt41_dir = "result/gpt41-balanced-full"
    results = {}
    
    print("📊 加载GPT-4.1结果...")
    for filename in os.listdir(gpt41_dir):
        if filename.startswith("actionSummary_") and filename.endswith(".json"):
            video_id = filename.replace("actionSummary_", "").replace(".json", "")
            
            with open(os.path.join(gpt41_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取key_actions
            key_actions = []
            for segment in data:
                if isinstance(segment, dict) and 'key_actions' in segment:
                    key_actions.append(segment['key_actions'])
            
            results[video_id] = key_actions
    
    print(f"✅ 加载了 {len(results)} 个GPT-4.1视频结果")
    return results

def load_ground_truth():
    """加载Ground Truth标签"""
    df = pd.read_csv('result/groundtruth_labels.csv', sep='\t')
    
    ground_truth = {}
    for _, row in df.iterrows():
        video_id = row['video_id'].replace('.avi', '')
        label = row['ground_truth_label']
        
        # 转换标签
        if label == 'none':
            ground_truth[video_id] = 'none'
        else:
            ground_truth[video_id] = 'ghost probing'
    
    print(f"✅ 加载了 {len(ground_truth)} 个Ground Truth标签")
    return ground_truth

def evaluate_model_results(results, ground_truth, model_name):
    """评估模型结果"""
    y_true = []
    y_pred = []
    details = []
    
    for video_id in sorted(results.keys()):
        if video_id in ground_truth:
            # Ground truth
            gt_label = ground_truth[video_id]
            y_true.append(gt_label)
            
            # 模型预测 - any segment策略
            predictions = results[video_id]
            has_ghost_probing = any('ghost probing' in str(pred).lower() for pred in predictions)
            
            if has_ghost_probing:
                pred_label = 'ghost probing'
            else:
                pred_label = 'none'
            
            y_pred.append(pred_label)
            
            details.append({
                'video_id': video_id,
                'ground_truth': gt_label,
                'prediction': pred_label,
                'correct': gt_label == pred_label,
                'raw_predictions': predictions
            })
    
    return y_true, y_pred, details

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    # 转换为二元分类
    y_true_binary = [1 if label == 'ghost probing' else 0 for label in y_true]
    y_pred_binary = [1 if label == 'ghost probing' else 0 for label in y_pred]
    
    # 混淆矩阵
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'confusion_matrix': cm,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

def create_comprehensive_comparison_plot(metrics_gemini, metrics_gpt41, save_path):
    """创建综合对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 指标对比柱状图
    metrics = ['precision', 'recall', 'f1', 'accuracy', 'specificity']
    gemini_values = [metrics_gemini[m] for m in metrics]
    gpt41_values = [metrics_gpt41[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, gemini_values, width, label='Gemini 2.0 Flash', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, gpt41_values, width, label='GPT-4.1 Balanced', alpha=0.8, color='orange')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Comparison (97 Videos)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.capitalize() for m in metrics])
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # 2. Gemini混淆矩阵
    sns.heatmap(metrics_gemini['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted None', 'Predicted Ghost Probing'],
                yticklabels=['Actual None', 'Actual Ghost Probing'],
                ax=ax2)
    ax2.set_title('Gemini 2.0 Flash - Confusion Matrix')
    
    # 3. GPT-4.1混淆矩阵
    sns.heatmap(metrics_gpt41['confusion_matrix'], annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Predicted None', 'Predicted Ghost Probing'],
                yticklabels=['Actual None', 'Actual Ghost Probing'],
                ax=ax3)
    ax3.set_title('GPT-4.1 Balanced - Confusion Matrix')
    
    # 4. F1分数对比
    f1_comparison = [metrics_gemini['f1'], metrics_gpt41['f1']]
    models = ['Gemini 2.0 Flash', 'GPT-4.1 Balanced']
    colors = ['skyblue', 'orange']
    
    bars = ax4.bar(models, f1_comparison, color=colors, alpha=0.8)
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score Comparison')
    ax4.set_ylim(0, 1)
    
    # 添加F1分数标签
    for bar, f1 in zip(bars, f1_comparison):
        height = bar.get_height()
        ax4.annotate(f'{f1:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 综合对比图已保存: {save_path}")

def analyze_disagreements(details_gemini, details_gpt41):
    """分析模型分歧案例"""
    disagreements = []
    
    # 创建视频ID到详情的映射
    gemini_dict = {d['video_id']: d for d in details_gemini}
    gpt41_dict = {d['video_id']: d for d in details_gpt41}
    
    common_videos = set(gemini_dict.keys()) & set(gpt41_dict.keys())
    
    for video_id in common_videos:
        gem_detail = gemini_dict[video_id]
        gpt_detail = gpt41_dict[video_id]
        
        if gem_detail['prediction'] != gpt_detail['prediction']:
            disagreements.append({
                'video_id': video_id,
                'ground_truth': gem_detail['ground_truth'],
                'gemini_pred': gem_detail['prediction'],
                'gpt41_pred': gpt_detail['prediction'],
                'gemini_correct': gem_detail['correct'],
                'gpt41_correct': gpt_detail['correct']
            })
    
    return disagreements

def main():
    print("🔍 Gemini vs GPT-4.1 完整对比分析 (97个视频)")
    print("=" * 60)
    
    # 加载数据
    gemini_results = load_gemini_results()
    gpt41_results = load_gpt41_results()
    ground_truth = load_ground_truth()
    
    # 找到共同视频
    common_videos = set(gemini_results.keys()) & set(gpt41_results.keys()) & set(ground_truth.keys())
    print(f"📹 共同视频: {len(common_videos)} 个")
    
    # 检查缺失的视频
    all_gt_videos = set(ground_truth.keys())
    missing_from_gemini = all_gt_videos - set(gemini_results.keys())
    missing_from_gpt41 = all_gt_videos - set(gpt41_results.keys())
    
    if missing_from_gemini:
        print(f"⚠️ Gemini缺失视频: {sorted(list(missing_from_gemini))}")
    if missing_from_gpt41:
        print(f"⚠️ GPT-4.1缺失视频: {sorted(list(missing_from_gpt41))}")
    
    # 创建子集数据
    gemini_subset = {vid: gemini_results[vid] for vid in common_videos}
    gpt41_subset = {vid: gpt41_results[vid] for vid in common_videos}
    gt_subset = {vid: ground_truth[vid] for vid in common_videos}
    
    print(f"\n🎯 对比分析 ({len(common_videos)} 个视频):") 
    print("-" * 50)
    
    # 评估两个模型
    y_true_gemini, y_pred_gemini, details_gemini = evaluate_model_results(gemini_subset, gt_subset, "Gemini")
    y_true_gpt41, y_pred_gpt41, details_gpt41 = evaluate_model_results(gpt41_subset, gt_subset, "GPT-4.1")
    
    # 计算指标
    metrics_gemini = calculate_metrics(y_true_gemini, y_pred_gemini)
    metrics_gpt41 = calculate_metrics(y_true_gpt41, y_pred_gpt41)
    
    # 打印详细结果
    print("🤖 Gemini 2.0 Flash + 平衡版Prompt:")
    print(f"  📊 精确度 (Precision): {metrics_gemini['precision']:.3f}")
    print(f"  📊 召回率 (Recall): {metrics_gemini['recall']:.3f}")
    print(f"  📊 F1分数: {metrics_gemini['f1']:.3f}")
    print(f"  📊 准确率 (Accuracy): {metrics_gemini['accuracy']:.3f}")
    print(f"  📊 特异性 (Specificity): {metrics_gemini['specificity']:.3f}")
    print(f"  📊 混淆矩阵: TP={metrics_gemini['tp']}, FP={metrics_gemini['fp']}, FN={metrics_gemini['fn']}, TN={metrics_gemini['tn']}")
    
    print("\n🤖 GPT-4.1 + 平衡版Prompt:")
    print(f"  📊 精确度 (Precision): {metrics_gpt41['precision']:.3f}")
    print(f"  📊 召回率 (Recall): {metrics_gpt41['recall']:.3f}")
    print(f"  📊 F1分数: {metrics_gpt41['f1']:.3f}")
    print(f"  📊 准确率 (Accuracy): {metrics_gpt41['accuracy']:.3f}")
    print(f"  📊 特异性 (Specificity): {metrics_gpt41['specificity']:.3f}")
    print(f"  📊 混淆矩阵: TP={metrics_gpt41['tp']}, FP={metrics_gpt41['fp']}, FN={metrics_gpt41['fn']}, TN={metrics_gpt41['tn']}")
    
    # 对比分析
    print(f"\n⚖️ 性能差异分析:")
    precision_diff = metrics_gemini['precision'] - metrics_gpt41['precision']
    recall_diff = metrics_gemini['recall'] - metrics_gpt41['recall']
    f1_diff = metrics_gemini['f1'] - metrics_gpt41['f1']
    accuracy_diff = metrics_gemini['accuracy'] - metrics_gpt41['accuracy']
    
    print(f"  🎯 精确度差异: {precision_diff:+.3f} {'(Gemini更好)' if precision_diff > 0 else '(GPT-4.1更好)' if precision_diff < 0 else '(相等)'}")
    print(f"  🎯 召回率差异: {recall_diff:+.3f} {'(Gemini更好)' if recall_diff > 0 else '(GPT-4.1更好)' if recall_diff < 0 else '(相等)'}")
    print(f"  🎯 F1分数差异: {f1_diff:+.3f} {'(Gemini更好)' if f1_diff > 0 else '(GPT-4.1更好)' if f1_diff < 0 else '(相等)'}")
    print(f"  🎯 准确率差异: {accuracy_diff:+.3f} {'(Gemini更好)' if accuracy_diff > 0 else '(GPT-4.1更好)' if accuracy_diff < 0 else '(相等)'}")
    
    # 分析分歧案例
    disagreements = analyze_disagreements(details_gemini, details_gpt41)
    consistent_count = len(common_videos) - len(disagreements)
    consistency_rate = consistent_count / len(common_videos) * 100
    
    print(f"\n📋 模型一致性分析:")
    print(f"  ✅ 一致预测: {consistent_count}/{len(common_videos)} ({consistency_rate:.1f}%)")
    print(f"  ❌ 分歧案例: {len(disagreements)} 个")
    
    # 显示部分分歧案例
    if disagreements:
        print(f"\n🔍 分歧案例详情 (前10个):")
        print(f"{'Video ID':<15} {'Ground Truth':<15} {'Gemini':<20} {'GPT-4.1':<20} {'谁对了':<10}")
        print("-" * 85)
        
        for i, disagreement in enumerate(disagreements[:10]):
            who_correct = ""
            if disagreement['gemini_correct'] and not disagreement['gpt41_correct']:
                who_correct = "Gemini"
            elif disagreement['gpt41_correct'] and not disagreement['gemini_correct']:
                who_correct = "GPT-4.1"
            elif disagreement['gemini_correct'] and disagreement['gpt41_correct']:
                who_correct = "都对"
            else:
                who_correct = "都错"
            
            print(f"{disagreement['video_id']:<15} {disagreement['ground_truth']:<15} {disagreement['gemini_pred']:<20} {disagreement['gpt41_pred']:<20} {who_correct:<10}")
    
    # 创建可视化
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"result/gemini_vs_gpt41_final_97videos_{timestamp}.png"
    create_comprehensive_comparison_plot(metrics_gemini, metrics_gpt41, plot_path)
    
    # 保存详细结果
    comparison_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_videos': len(common_videos),
        'missing_videos': {
            'gemini_missing': sorted(list(missing_from_gemini)),
            'gpt41_missing': sorted(list(missing_from_gpt41))
        },
        'gemini_metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics_gemini.items()},
        'gpt41_metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics_gpt41.items()},
        'consistency_rate': consistency_rate,
        'disagreements': disagreements,
        'common_videos': sorted(list(common_videos)),
        'summary': {
            'better_precision': 'Gemini' if precision_diff > 0 else 'GPT-4.1' if precision_diff < 0 else 'Tie',
            'better_recall': 'Gemini' if recall_diff > 0 else 'GPT-4.1' if recall_diff < 0 else 'Tie',
            'better_f1': 'Gemini' if f1_diff > 0 else 'GPT-4.1' if f1_diff < 0 else 'Tie',
            'better_accuracy': 'Gemini' if accuracy_diff > 0 else 'GPT-4.1' if accuracy_diff < 0 else 'Tie'
        }
    }
    
    output_file = f"result/gemini_vs_gpt41_final_97videos_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 完整对比结果已保存:")
    print(f"  📊 详细数据: {output_file}")
    print(f"  📈 对比图表: {plot_path}")
    
    # 总结
    print(f"\n🎯 最终总结:")
    winner_count = {'Gemini': 0, 'GPT-4.1': 0, 'Tie': 0}
    for metric in ['better_precision', 'better_recall', 'better_f1', 'better_accuracy']:
        winner_count[comparison_results['summary'][metric]] += 1
    
    print(f"  🏆 各指标胜负: Gemini {winner_count['Gemini']} : {winner_count['GPT-4.1']} GPT-4.1 (平局: {winner_count['Tie']})")
    
    if winner_count['Gemini'] > winner_count['GPT-4.1']:
        print(f"  🎉 总体表现: Gemini 2.0 Flash 更优")
    elif winner_count['GPT-4.1'] > winner_count['Gemini']:
        print(f"  🎉 总体表现: GPT-4.1 更优")
    else:
        print(f"  🤝 总体表现: 两模型表现相当")
    
    print(f"\n📊 数据完整性:")
    print(f"  ✅ Gemini处理: {len(gemini_results)}/99 ({len(gemini_results)/99*100:.1f}%)")
    print(f"  ✅ GPT-4.1处理: {len(gpt41_results)}/99 ({len(gpt41_results)/99*100:.1f}%)")
    print(f"  📋 共同对比: {len(common_videos)}/99 ({len(common_videos)/99*100:.1f}%)")
    
    print(f"\n✅ 97视频完整对比分析完成！")

if __name__ == "__main__":
    main()