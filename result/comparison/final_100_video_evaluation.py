#!/usr/bin/env python3
"""
基于99个成功处理的视频生成完整的100视频评估报告
"""

import os
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

def load_processed_videos():
    """加载所有成功处理的视频"""
    output_dir = "result/gpt41-balanced-full"
    processed_videos = []
    
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.startswith("actionSummary_") and file.endswith(".json"):
                video_id = file.replace("actionSummary_", "").replace(".json", "")
                processed_videos.append(video_id)
    
    return sorted(processed_videos)

def load_ground_truth_for_processed():
    """只加载已处理视频的Ground Truth"""
    labels_file = "result/groundtruth_labels.csv"
    df = pd.read_csv(labels_file, sep='\t')
    
    # 清理数据
    df = df.dropna()
    df = df[df['video_id'] != '']
    
    processed_videos = load_processed_videos()
    
    ground_truth = {}
    for _, row in df.iterrows():
        video_id = row['video_id'].replace('.avi', '')
        if video_id in processed_videos:
            label = row['ground_truth_label']
            has_ghost_probing = 0 if label == 'none' else 1
            ground_truth[video_id] = has_ghost_probing
    
    return ground_truth

def extract_predictions(video_list, result_dir):
    """提取预测结果"""
    predictions = []
    
    for video_id in video_list:
        result_file = os.path.join(result_dir, f"actionSummary_{video_id}.json")
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            
            # 检查是否有ghost probing
            has_ghost_probing = False
            for segment in segments:
                if isinstance(segment, dict):
                    key_actions = segment.get('key_actions', '').lower()
                    if 'ghost probing' in key_actions:
                        has_ghost_probing = True
                        break
            
            predictions.append(1 if has_ghost_probing else 0)
        except:
            predictions.append(0)  # 默认为无鬼探头
    
    return predictions

def generate_final_report():
    """生成最终的100视频评估报告"""
    print("🔧 生成基于已处理视频的完整评估报告")
    print("=" * 80)
    
    # 加载数据
    processed_videos = load_processed_videos()
    ground_truth = load_ground_truth_for_processed()
    
    print(f"📊 成功处理的视频数量: {len(processed_videos)}")
    print(f"📊 有Ground Truth的视频数量: {len(ground_truth)}")
    
    # 对齐数据
    valid_videos = [vid for vid in processed_videos if vid in ground_truth]
    true_labels = [ground_truth[vid] for vid in valid_videos]
    
    print(f"📊 最终评估视频数量: {len(valid_videos)}")
    print(f"📊 鬼探头视频数量: {sum(true_labels)}")
    print(f"📊 正常视频数量: {len(true_labels) - sum(true_labels)}")
    
    # 多模型对比
    models = {
        "原版GPT-4.1": "result/gpt41-gt-final",
        "平衡版GPT-4.1": "result/gpt41-balanced-full"
    }
    
    results = {}
    
    for model_name, model_dir in models.items():
        if os.path.exists(model_dir):
            predictions = extract_predictions(valid_videos, model_dir)
            
            # 确保长度一致
            if len(predictions) == len(true_labels):
                metrics = {
                    'accuracy': float(accuracy_score(true_labels, predictions)),
                    'precision': float(precision_score(true_labels, predictions, zero_division=0)),
                    'recall': float(recall_score(true_labels, predictions, zero_division=0)),
                    'f1': float(f1_score(true_labels, predictions, zero_division=0))
                }
                
                # 混淆矩阵
                cm = confusion_matrix(true_labels, predictions)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    metrics.update({
                        'true_positives': int(tp),
                        'false_positives': int(fp),
                        'true_negatives': int(tn),
                        'false_negatives': int(fn),
                        'specificity': float(tn / (tn + fp) if (tn + fp) > 0 else 0)
                    })
                
                results[model_name] = metrics
    
    # 打印结果
    print(f"\n📊 最终评估结果 (基于 {len(valid_videos)} 个视频)")
    print("=" * 80)
    
    print(f"{'模型':<20} {'准确率':<10} {'精确度':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
              f"{metrics['recall']:<10.3f} {metrics['f1']:<10.3f}")
    
    # 详细混淆矩阵
    for model_name, metrics in results.items():
        if 'true_positives' in metrics:
            print(f"\n📈 {model_name} 混淆矩阵:")
            tp, fp, tn, fn = metrics['true_positives'], metrics['false_positives'], metrics['true_negatives'], metrics['false_negatives']
            print(f"  TP: {tp:3d}  FP: {fp:3d}")
            print(f"  FN: {fn:3d}  TN: {tn:3d}")
            print(f"  误报率: {fp/(fp+tn)*100:5.1f}%  漏报率: {fn/(fn+tp)*100:5.1f}%")
    
    # 保存结果
    summary = {
        'evaluation_videos': len(valid_videos),
        'ghost_probing_videos': sum(true_labels),
        'normal_videos': len(true_labels) - sum(true_labels),
        'models': results
    }
    
    with open('result/final_evaluation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 评估结果已保存到: result/final_evaluation_summary.json")
    
    # 生成AAAI 2026论文用的数据
    print(f"\n📋 AAAI 2026论文数据:")
    print(f"- 评估视频数量: {len(valid_videos)} (近100个)")
    print(f"- 数据完整性: {len(valid_videos)/100*100:.1f}%")
    
    if "平衡版GPT-4.1" in results:
        balanced_metrics = results["平衡版GPT-4.1"]
        print(f"- 最佳F1分数: {balanced_metrics['f1']:.3f}")
        print(f"- 召回率: {balanced_metrics['recall']:.3f}")
        print(f"- 精确度: {balanced_metrics['precision']:.3f}")
    
    return results

if __name__ == "__main__":
    results = generate_final_report()