#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini 2.0 Flash Prompt改进效果分析
对比使用平衡版prompt前后的性能变化
"""

import os
import json
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_gemini_balanced_results():
    """加载使用平衡版prompt的Gemini 2.0 Flash结果"""
    gemini_dir = "result/gemini-balanced-full"
    results = {}
    
    print("📊 加载Gemini 2.0 Flash + 平衡版Prompt结果...")
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
    
    print(f"✅ 加载了 {len(results)} 个平衡版Prompt结果")
    return results

def load_gemini_original_results():
    """尝试加载原始prompt的Gemini结果"""
    # 检查可能的原始结果目录
    possible_dirs = [
        "result/gemini-1.5-flash",
        "result/gemini-2.0-flash-original", 
        "result/gemini-original",
        "result/gemini-baseline"
    ]
    
    results = {}
    found_dir = None
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            print(f"📊 检查目录: {dir_path}")
            file_count = len([f for f in os.listdir(dir_path) if f.startswith("actionSummary_") and f.endswith(".json")])
            print(f"  找到 {file_count} 个结果文件")
            
            if file_count > 10:  # 如果有足够多的文件
                found_dir = dir_path
                break
    
    if not found_dir:
        print("❌ 未找到原始prompt的Gemini结果")
        return None, None
    
    print(f"📊 加载原始Prompt结果从: {found_dir}")
    
    # 确定模型类型
    model_type = "Gemini 1.5 Flash" if "1.5" in found_dir else "Gemini 2.0 Flash"
    
    for filename in os.listdir(found_dir):
        if filename.startswith("actionSummary_") and filename.endswith(".json"):
            video_id = filename.replace("actionSummary_", "").replace(".json", "")
            
            try:
                with open(os.path.join(found_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 提取key_actions
                key_actions = []
                for segment in data:
                    if isinstance(segment, dict) and 'key_actions' in segment:
                        key_actions.append(segment['key_actions'])
                
                results[video_id] = key_actions
            except Exception as e:
                continue
    
    print(f"✅ 加载了 {len(results)} 个原始Prompt结果 ({model_type})")
    return results, model_type

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

def create_improvement_comparison_plot(metrics_original, metrics_balanced, original_model_type, save_path):
    """创建prompt改进效果对比图"""
    metrics = ['precision', 'recall', 'f1', 'accuracy', 'specificity']
    original_values = [metrics_original[m] for m in metrics]
    balanced_values = [metrics_balanced[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, original_values, width, label=f'{original_model_type} (原始Prompt)', alpha=0.8, color='lightcoral')
    bars2 = ax.bar(x + width/2, balanced_values, width, label='Gemini 2.0 Flash (平衡版Prompt)', alpha=0.8, color='skyblue')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    # 添加改进箭头和百分比
    for i, (orig, bal) in enumerate(zip(original_values, balanced_values)):
        improvement = ((bal - orig) / orig * 100) if orig > 0 else 0
        if abs(improvement) > 1:  # 只显示显著改进
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'{improvement:+.1f}%', 
                       xy=(i, max(orig, bal) + 0.05),
                       ha='center', va='bottom',
                       color=color, fontweight='bold')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Gemini Prompt改进效果对比')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 改进效果对比图已保存: {save_path}")

def main():
    print("🔍 Gemini 2.0 Flash Prompt改进效果分析")
    print("=" * 50)
    
    # 加载数据
    gemini_balanced = load_gemini_balanced_results()
    gemini_original, original_model_type = load_gemini_original_results()
    
    if not gemini_original:
        print("❌ 无法找到原始prompt的结果进行对比")
        print("💡 建议: 如果有原始结果，请确保放在以下目录之一:")
        print("  - result/gemini-2.0-flash-original/")
        print("  - result/gemini-original/") 
        print("  - result/gemini-baseline/")
        return
    
    ground_truth = load_ground_truth()
    
    # 找到共同视频
    common_videos = set(gemini_balanced.keys()) & set(gemini_original.keys()) & set(ground_truth.keys())
    print(f"📹 可对比的共同视频: {len(common_videos)} 个")
    
    if len(common_videos) < 5:
        print("❌ 共同视频太少，无法进行有效对比")
        return
    
    # 创建子集数据
    balanced_subset = {vid: gemini_balanced[vid] for vid in common_videos}
    original_subset = {vid: gemini_original[vid] for vid in common_videos}
    gt_subset = {vid: ground_truth[vid] for vid in common_videos}
    
    print(f"\n🎯 对比分析 ({len(common_videos)} 个视频):")
    print("-" * 50)
    
    # 评估两个版本
    y_true_original, y_pred_original, details_original = evaluate_model_results(original_subset, gt_subset, original_model_type)
    y_true_balanced, y_pred_balanced, details_balanced = evaluate_model_results(balanced_subset, gt_subset, "Gemini 2.0 Flash + 平衡版Prompt")
    
    # 计算指标
    metrics_original = calculate_metrics(y_true_original, y_pred_original)
    metrics_balanced = calculate_metrics(y_true_balanced, y_pred_balanced)
    
    # 打印详细结果
    print(f"🤖 {original_model_type} + 原始Prompt:")
    print(f"  📊 精确度 (Precision): {metrics_original['precision']:.3f}")
    print(f"  📊 召回率 (Recall): {metrics_original['recall']:.3f}")
    print(f"  📊 F1分数: {metrics_original['f1']:.3f}")
    print(f"  📊 准确率 (Accuracy): {metrics_original['accuracy']:.3f}")
    print(f"  📊 特异性 (Specificity): {metrics_original['specificity']:.3f}")
    print(f"  📊 混淆矩阵: TP={metrics_original['tp']}, FP={metrics_original['fp']}, FN={metrics_original['fn']}, TN={metrics_original['tn']}")
    
    print(f"\n🤖 Gemini 2.0 Flash + 平衡版Prompt:")
    print(f"  📊 精确度 (Precision): {metrics_balanced['precision']:.3f}")
    print(f"  📊 召回率 (Recall): {metrics_balanced['recall']:.3f}")
    print(f"  📊 F1分数: {metrics_balanced['f1']:.3f}")
    print(f"  📊 准确率 (Accuracy): {metrics_balanced['accuracy']:.3f}")
    print(f"  📊 特异性 (Specificity): {metrics_balanced['specificity']:.3f}")
    print(f"  📊 混淆矩阵: TP={metrics_balanced['tp']}, FP={metrics_balanced['fp']}, FN={metrics_balanced['fn']}, TN={metrics_balanced['tn']}")
    
    # 改进分析
    print(f"\n📈 Prompt改进效果:")
    improvements = {}
    for metric in ['precision', 'recall', 'f1', 'accuracy', 'specificity']:
        old_val = metrics_original[metric]
        new_val = metrics_balanced[metric]
        change = new_val - old_val
        pct_change = (change / old_val * 100) if old_val > 0 else 0
        improvements[metric] = {
            'absolute': change,
            'percentage': pct_change,
            'improved': change > 0
        }
        
        status = "📈 改进" if change > 0 else "📉 下降" if change < 0 else "➡️ 持平"
        print(f"  {metric.capitalize()}: {old_val:.3f} → {new_val:.3f} ({change:+.3f}, {pct_change:+.1f}%) {status}")
    
    # 总体评估
    improved_count = sum(1 for imp in improvements.values() if imp['improved'])
    total_metrics = len(improvements)
    
    print(f"\n🎯 总体改进效果:")
    print(f"  📊 改进指标: {improved_count}/{total_metrics}")
    print(f"  📈 改进率: {improved_count/total_metrics*100:.1f}%")
    
    if improved_count > total_metrics / 2:
        print(f"  🎉 总体评价: 平衡版Prompt显著改进了性能")
    elif improved_count == total_metrics / 2:
        print(f"  🤝 总体评价: 平衡版Prompt效果持平")
    else:
        print(f"  📉 总体评价: 平衡版Prompt对某些指标有负面影响")
    
    # 创建可视化
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"result/gemini_prompt_improvement_{timestamp}.png"
    create_improvement_comparison_plot(metrics_original, metrics_balanced, original_model_type, plot_path)
    
    # 保存详细结果
    comparison_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'common_videos': len(common_videos),
        'original_model': original_model_type,
        'metrics_original': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics_original.items()},
        'metrics_balanced': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics_balanced.items()},
        'improvements': improvements,
        'summary': {
            'improved_metrics': improved_count,
            'total_metrics': total_metrics,
            'improvement_rate': improved_count/total_metrics*100
        }
    }
    
    output_file = f"result/gemini_prompt_improvement_analysis_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 改进分析结果已保存:")
    print(f"  📊 详细数据: {output_file}")
    print(f"  📈 对比图表: {plot_path}")
    
    print(f"\n✅ Prompt改进效果分析完成！")

if __name__ == "__main__":
    main()