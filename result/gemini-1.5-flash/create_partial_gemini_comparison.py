#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于已完成的12个Gemini视频结果创建部分对比分析
"""

import os
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def load_gemini_results():
    """加载Gemini结果"""
    gemini_dir = "result/gemini-balanced-full"
    results = {}
    
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
    
    return results

def load_gpt41_results():
    """加载GPT-4.1平衡版结果"""
    gpt41_dir = "result/gpt41-balanced-full"
    results = {}
    
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
    
    return ground_truth

def evaluate_model_results(results, ground_truth, model_name):
    """评估模型结果"""
    y_true = []
    y_pred = []
    
    for video_id in results:
        if video_id in ground_truth:
            # Ground truth
            gt_label = ground_truth[video_id]
            y_true.append(gt_label)
            
            # 模型预测 - any segment策略
            predictions = results[video_id]
            has_ghost_probing = any('ghost probing' in str(pred).lower() for pred in predictions)
            
            if has_ghost_probing:
                y_pred.append('ghost probing')
            else:
                y_pred.append('none')
    
    return y_true, y_pred

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
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

def main():
    print("🔍 Gemini vs GPT-4.1 部分对比分析")
    print("=" * 60)
    print("📋 基于已完成的12个Gemini视频结果进行对比")
    
    # 加载数据
    print("\n📊 加载数据...")
    gemini_results = load_gemini_results()
    gpt41_results = load_gpt41_results()
    ground_truth = load_ground_truth()
    
    print(f"✅ Gemini结果: {len(gemini_results)} 个视频")
    print(f"✅ GPT-4.1结果: {len(gpt41_results)} 个视频")
    print(f"✅ Ground Truth: {len(ground_truth)} 个标签")
    
    # 找到共同视频
    common_videos = set(gemini_results.keys()) & set(gpt41_results.keys()) & set(ground_truth.keys())
    print(f"📹 共同视频: {len(common_videos)} 个")
    
    if len(common_videos) == 0:
        print("❌ 没有找到共同的视频进行对比")
        return
    
    # 创建子集数据
    gemini_subset = {vid: gemini_results[vid] for vid in common_videos}
    gpt41_subset = {vid: gpt41_results[vid] for vid in common_videos}
    gt_subset = {vid: ground_truth[vid] for vid in common_videos}
    
    print(f"\n🎯 对比分析 ({len(common_videos)} 个视频):")
    print("-" * 40)
    
    # 评估Gemini
    y_true_gemini, y_pred_gemini = evaluate_model_results(gemini_subset, gt_subset, "Gemini")
    metrics_gemini = calculate_metrics(y_true_gemini, y_pred_gemini)
    
    # 评估GPT-4.1
    y_true_gpt41, y_pred_gpt41 = evaluate_model_results(gpt41_subset, gt_subset, "GPT-4.1")
    metrics_gpt41 = calculate_metrics(y_true_gpt41, y_pred_gpt41)
    
    # 打印结果
    print("🤖 Gemini 2.0 Flash + 平衡版Prompt:")
    print(f"  📊 精确度: {metrics_gemini['precision']:.3f}")
    print(f"  📊 召回率: {metrics_gemini['recall']:.3f}")
    print(f"  📊 F1分数: {metrics_gemini['f1']:.3f}")
    print(f"  📊 准确率: {metrics_gemini['accuracy']:.3f}")
    print(f"  📊 混淆矩阵: TP={metrics_gemini['tp']}, FP={metrics_gemini['fp']}, FN={metrics_gemini['fn']}, TN={metrics_gemini['tn']}")
    
    print("\n🤖 GPT-4.1 + 平衡版Prompt:")
    print(f"  📊 精确度: {metrics_gpt41['precision']:.3f}")
    print(f"  📊 召回率: {metrics_gpt41['recall']:.3f}")
    print(f"  📊 F1分数: {metrics_gpt41['f1']:.3f}")
    print(f"  📊 准确率: {metrics_gpt41['accuracy']:.3f}")
    print(f"  📊 混淆矩阵: TP={metrics_gpt41['tp']}, FP={metrics_gpt41['fp']}, FN={metrics_gpt41['fn']}, TN={metrics_gpt41['tn']}")
    
    # 对比分析
    print(f"\n⚖️ 对比分析:")
    print(f"  🎯 精确度差异: {(metrics_gemini['precision'] - metrics_gpt41['precision']):.3f}")
    print(f"  🎯 召回率差异: {(metrics_gemini['recall'] - metrics_gpt41['recall']):.3f}")
    print(f"  🎯 F1分数差异: {(metrics_gemini['f1'] - metrics_gpt41['f1']):.3f}")
    
    # 详细对比
    print(f"\n📋 视频详细对比:")
    print(f"{'Video ID':<15} {'Ground Truth':<15} {'Gemini':<20} {'GPT-4.1':<20} {'一致性':<8}")
    print("-" * 80)
    
    consistent_count = 0
    for i, video_id in enumerate(sorted(common_videos)):
        gt = y_true_gemini[i] if i < len(y_true_gemini) else 'N/A'
        gemini_pred = y_pred_gemini[i] if i < len(y_pred_gemini) else 'N/A'
        gpt41_pred = y_pred_gpt41[i] if i < len(y_pred_gpt41) else 'N/A'
        
        consistent = "✅" if gemini_pred == gpt41_pred else "❌"
        if gemini_pred == gpt41_pred:
            consistent_count += 1
        
        print(f"{video_id:<15} {gt:<15} {gemini_pred:<20} {gpt41_pred:<20} {consistent:<8}")
    
    consistency_rate = consistent_count / len(common_videos) * 100
    print(f"\n🎯 模型一致性: {consistent_count}/{len(common_videos)} ({consistency_rate:.1f}%)")
    
    # 保存结果
    comparison_results = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_videos': len(common_videos),
        'gemini_metrics': metrics_gemini,
        'gpt41_metrics': metrics_gpt41,
        'consistency_rate': consistency_rate,
        'video_details': {
            'common_videos': list(common_videos),
            'ground_truth': y_true_gemini,
            'gemini_predictions': y_pred_gemini,
            'gpt41_predictions': y_pred_gpt41
        }
    }
    
    # 转换numpy类型为Python原生类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    comparison_results = convert_numpy_types(comparison_results)
    
    output_file = f"result/gemini_vs_gpt41_partial_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 对比结果已保存: {output_file}")
    print(f"\n🎯 部分对比完成！等待完整99视频实验完成后进行全面分析。")

if __name__ == "__main__":
    main()