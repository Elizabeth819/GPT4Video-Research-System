#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini Prompt改进效果简化分析 - 修复版
"""

import os
import json
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

def load_gemini_balanced_results():
    """加载使用平衡版prompt的Gemini 2.0 Flash结果"""
    gemini_dir = "result/gemini-balanced-full"
    results = {}
    
    for filename in os.listdir(gemini_dir):
        if filename.startswith("actionSummary_") and filename.endswith(".json"):
            video_id = filename.replace("actionSummary_", "").replace(".json", "")
            if video_id.startswith("dada_"):
                video_id = video_id.replace("dada_", "images_")
            
            with open(os.path.join(gemini_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            key_actions = []
            for segment in data:
                if isinstance(segment, dict) and 'key_actions' in segment:
                    key_actions.append(segment['key_actions'])
            
            results[video_id] = key_actions
    
    return results

def load_gemini_original_results():
    """加载原始prompt的Gemini 1.5 Flash结果"""
    gemini_dir = "result/gemini-1.5-flash"
    results = {}
    
    for filename in os.listdir(gemini_dir):
        if filename.startswith("actionSummary_") and filename.endswith(".json"):
            video_id = filename.replace("actionSummary_", "").replace(".json", "")
            
            try:
                with open(os.path.join(gemini_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                key_actions = []
                for segment in data:
                    if isinstance(segment, dict) and 'key_actions' in segment:
                        key_actions.append(segment['key_actions'])
                
                results[video_id] = key_actions
            except:
                continue
    
    return results

def load_ground_truth():
    """加载Ground Truth标签"""
    df = pd.read_csv('result/groundtruth_labels.csv', sep='\t')
    ground_truth = {}
    
    for _, row in df.iterrows():
        video_id = row['video_id'].replace('.avi', '')
        label = row['ground_truth_label']
        ground_truth[video_id] = 'none' if label == 'none' else 'ghost probing'
    
    return ground_truth

def evaluate_results(results, ground_truth):
    """评估结果"""
    y_true = []
    y_pred = []
    
    for video_id in sorted(results.keys()):
        if video_id in ground_truth:
            gt_label = ground_truth[video_id]
            y_true.append(gt_label)
            
            predictions = results[video_id]
            has_ghost_probing = any('ghost probing' in str(pred).lower() for pred in predictions)
            pred_label = 'ghost probing' if has_ghost_probing else 'none'
            y_pred.append(pred_label)
    
    return y_true, y_pred

def calculate_metrics(y_true, y_pred):
    """计算指标"""
    y_true_binary = [1 if label == 'ghost probing' else 0 for label in y_true]
    y_pred_binary = [1 if label == 'ghost probing' else 0 for label in y_pred]
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
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
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

def main():
    print("🔍 Gemini Prompt改进效果分析")
    print("=" * 50)
    
    # 加载数据
    balanced_results = load_gemini_balanced_results()
    original_results = load_gemini_original_results()
    ground_truth = load_ground_truth()
    
    print(f"📊 Gemini 2.0 Flash + 平衡版Prompt: {len(balanced_results)} 个视频")
    print(f"📊 Gemini 1.5 Flash + 原始Prompt: {len(original_results)} 个视频")
    
    # 找到共同视频
    common_videos = set(balanced_results.keys()) & set(original_results.keys()) & set(ground_truth.keys())
    print(f"📹 可对比的共同视频: {len(common_videos)} 个")
    
    # 评估
    balanced_subset = {vid: balanced_results[vid] for vid in common_videos}
    original_subset = {vid: original_results[vid] for vid in common_videos}
    gt_subset = {vid: ground_truth[vid] for vid in common_videos}
    
    y_true_orig, y_pred_orig = evaluate_results(original_subset, gt_subset)
    y_true_bal, y_pred_bal = evaluate_results(balanced_subset, gt_subset)
    
    metrics_original = calculate_metrics(y_true_orig, y_pred_orig)
    metrics_balanced = calculate_metrics(y_true_bal, y_pred_bal)
    
    print(f"\n🎯 对比结果 ({len(common_videos)} 个视频):")
    print("-" * 50)
    
    print("🤖 Gemini 1.5 Flash + 原始Prompt:")
    print(f"  📊 精确度: {metrics_original['precision']:.3f}")
    print(f"  📊 召回率: {metrics_original['recall']:.3f}")
    print(f"  📊 F1分数: {metrics_original['f1']:.3f}")
    print(f"  📊 准确率: {metrics_original['accuracy']:.3f}")
    print(f"  📊 特异性: {metrics_original['specificity']:.3f}")
    
    print("\n🤖 Gemini 2.0 Flash + 平衡版Prompt:")
    print(f"  📊 精确度: {metrics_balanced['precision']:.3f}")
    print(f"  📊 召回率: {metrics_balanced['recall']:.3f}")
    print(f"  📊 F1分数: {metrics_balanced['f1']:.3f}")
    print(f"  📊 准确率: {metrics_balanced['accuracy']:.3f}")
    print(f"  📊 特异性: {metrics_balanced['specificity']:.3f}")
    
    print(f"\n📈 改进效果:")
    improvements = {}
    for metric in ['precision', 'recall', 'f1', 'accuracy', 'specificity']:
        old_val = metrics_original[metric]
        new_val = metrics_balanced[metric]
        change = new_val - old_val
        pct_change = (change / old_val * 100) if old_val > 0 else 0
        
        status = "📈 改进" if change > 0 else "📉 下降" if change < 0 else "➡️ 持平"
        print(f"  {metric.capitalize()}: {old_val:.3f} → {new_val:.3f} ({change:+.3f}, {pct_change:+.1f}%) {status}")
        
        improvements[metric] = change > 0
    
    improved_count = sum(improvements.values())
    print(f"\n🎯 总结:")
    print(f"  📊 改进指标: {improved_count}/5")
    print(f"  📈 改进率: {improved_count/5*100:.1f}%")
    
    # 重要发现
    print(f"\n💡 重要发现:")
    print(f"  🔄 模型升级: Gemini 1.5 Flash → Gemini 2.0 Flash")
    print(f"  📝 Prompt优化: 原始Prompt → 平衡版Prompt")
    print(f"  🎯 精确度提升: {metrics_balanced['precision'] - metrics_original['precision']:+.3f}")
    print(f"  📉 召回率变化: {metrics_balanced['recall'] - metrics_original['recall']:+.3f}")
    print(f"  ⚖️ 特异性提升: {metrics_balanced['specificity'] - metrics_original['specificity']:+.3f}")
    
    # 策略分析
    if metrics_balanced['precision'] > metrics_original['precision'] and metrics_balanced['specificity'] > metrics_original['specificity']:
        print(f"\n📊 策略变化: 从高召回策略转向平衡策略")
        print(f"  ✅ 减少误报 (假阳性)")
        print(f"  ⚠️ 略微增加漏报 (假阴性)")
        print(f"  🎯 更适合精确检测场景")

if __name__ == "__main__":
    main()