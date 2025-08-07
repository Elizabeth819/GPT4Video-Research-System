#!/usr/bin/env python3
"""
精确的VideoChat2评估脚本
重新仔细对比VideoChat2结果与ground truth，确保统计指标准确性
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re

def load_ground_truth_labels(csv_path: str) -> Dict[str, str]:
    """加载ground truth标签"""
    ground_truth = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            if not row['video_id']:  # 跳过空行
                continue
                
            video_id = row['video_id'].replace('.avi', '')
            label = row['ground_truth_label']
            
            # 转换为二分类
            if label == 'none':
                ground_truth[video_id] = 'normal'
            elif 'ghost probing' in label:
                ground_truth[video_id] = 'ghost_probing'
            elif label == 'cut-in' or 'cut-in' in label:
                ground_truth[video_id] = 'normal'  # cut-in视为正常交通
            # 跳过其他不明确的标签
    
    return ground_truth

def extract_videochat2_predictions(results_dir: str) -> Dict[str, Dict]:
    """提取VideoChat2的预测结果和详细信息"""
    predictions = {}
    results_path = Path(results_dir)
    
    for json_file in results_path.glob("actionSummary_images_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                segment = data[0]
                
                # 从文件名提取video ID
                filename = json_file.stem
                match = re.search(r'actionSummary_images_(\d+)_(\d+)', filename)
                if match:
                    category, number = match.groups()
                    video_id = f"images_{category}_{number}"
                    
                    # 获取VideoChat2的分类信息
                    sentiment = segment.get('sentiment', '')
                    scene_theme = segment.get('scene_theme', '')
                    key_actions = segment.get('key_actions', '')
                    summary = segment.get('summary', '')
                    
                    # VideoChat2的分类逻辑：
                    # Negative + Dramatic + "ghost probing" in key_actions = ghost_probing
                    # Positive + Routine = normal
                    if (sentiment == 'Negative' and 
                        scene_theme == 'Dramatic' and 
                        'ghost probing' in key_actions.lower()):
                        prediction = 'ghost_probing'
                    else:
                        prediction = 'normal'
                    
                    predictions[video_id] = {
                        'prediction': prediction,
                        'sentiment': sentiment,
                        'scene_theme': scene_theme,
                        'key_actions': key_actions,
                        'summary': summary[:100] + '...' if len(summary) > 100 else summary,
                        'file': str(json_file)
                    }
                        
        except Exception as e:
            print(f"处理文件错误 {json_file}: {e}")
            continue
    
    return predictions

def calculate_accurate_metrics(ground_truth: Dict[str, str], predictions: Dict[str, Dict]) -> Dict:
    """计算准确的性能指标"""
    
    # 找到共同的video ID
    common_ids = set(ground_truth.keys()) & set(predictions.keys())
    
    if not common_ids:
        raise ValueError("ground truth和predictions之间没有找到共同的video ID")
    
    print(f"共同评估的视频数量: {len(common_ids)}")
    
    # 初始化混淆矩阵
    tp = fp = tn = fn = 0
    
    # 详细记录
    correct_predictions = []
    incorrect_predictions = []
    
    # 错误分类的详细信息
    false_positives = []  # 预测为ghost_probing但实际为normal
    false_negatives = []  # 预测为normal但实际为ghost_probing
    
    for video_id in sorted(common_ids):
        gt_label = ground_truth[video_id]
        pred_info = predictions[video_id]
        pred_label = pred_info['prediction']
        
        if gt_label == 'ghost_probing' and pred_label == 'ghost_probing':
            tp += 1
            correct_predictions.append((video_id, gt_label, pred_label, "True Positive"))
        elif gt_label == 'normal' and pred_label == 'ghost_probing':
            fp += 1
            incorrect_predictions.append((video_id, gt_label, pred_label, "False Positive"))
            false_positives.append({
                'video_id': video_id,
                'ground_truth': gt_label,
                'prediction': pred_label,
                'key_actions': pred_info['key_actions'],
                'sentiment': pred_info['sentiment'],
                'scene_theme': pred_info['scene_theme']
            })
        elif gt_label == 'normal' and pred_label == 'normal':
            tn += 1
            correct_predictions.append((video_id, gt_label, pred_label, "True Negative"))
        elif gt_label == 'ghost_probing' and pred_label == 'normal':
            fn += 1
            incorrect_predictions.append((video_id, gt_label, pred_label, "False Negative"))
            false_negatives.append({
                'video_id': video_id,
                'ground_truth': gt_label,
                'prediction': pred_label,
                'key_actions': pred_info['key_actions'],
                'sentiment': pred_info['sentiment'],
                'scene_theme': pred_info['scene_theme']
            })
    
    # 计算指标
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        },
        'confusion_matrix': {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'total_samples': total
        },
        'predictions': {
            'correct': correct_predictions,
            'incorrect': incorrect_predictions
        },
        'error_analysis': {
            'false_positives': false_positives,
            'false_negatives': false_negatives
        },
        'common_ids': sorted(list(common_ids))
    }

def print_detailed_report(results: Dict, ground_truth: Dict, predictions: Dict):
    """打印详细的评估报告"""
    
    metrics = results['metrics']
    cm = results['confusion_matrix']
    errors = results['error_analysis']
    
    print("=" * 80)
    print("VIDEOCHAT2 准确评估报告")
    print("=" * 80)
    
    # 性能指标
    print(f"\n📊 性能指标:")
    print(f"准确率 (Accuracy):  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"精确率 (Precision): {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"召回率 (Recall):    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"F1分数 (F1-Score):  {metrics['f1_score']:.3f}")
    
    # 混淆矩阵
    print(f"\n📈 混淆矩阵:")
    print("                     预测结果")
    print("                Ghost    Normal")
    print(f"实际 Ghost     {cm['true_positives']:5d}    {cm['false_negatives']:5d}")
    print(f"     Normal    {cm['false_positives']:5d}    {cm['true_negatives']:5d}")
    
    # Ground truth分布
    gt_ghost = sum(1 for label in ground_truth.values() if label == 'ghost_probing')
    gt_normal = sum(1 for label in ground_truth.values() if label == 'normal')
    
    print(f"\n📋 数据分布:")
    print(f"Ground Truth - 鬼探头: {gt_ghost}, 正常交通: {gt_normal}")
    
    # VideoChat2预测分布
    pred_ghost = sum(1 for pred_info in predictions.values() if pred_info['prediction'] == 'ghost_probing')
    pred_normal = sum(1 for pred_info in predictions.values() if pred_info['prediction'] == 'normal')
    
    print(f"VideoChat2预测 - 鬼探头: {pred_ghost}, 正常交通: {pred_normal}")
    
    # 错误分析
    print(f"\n❌ 错误分析:")
    
    if errors['false_positives']:
        print(f"\n假阳性 (False Positives): {len(errors['false_positives'])} 个视频")
        print("(VideoChat2错误地将正常交通识别为鬼探头)")
        for fp in errors['false_positives']:
            print(f"  {fp['video_id']}: GT={fp['ground_truth']}, 预测={fp['prediction']}")
            print(f"    - sentiment: {fp['sentiment']}, scene_theme: {fp['scene_theme']}")
            print(f"    - key_actions: {fp['key_actions']}")
    
    if errors['false_negatives']:
        print(f"\n假阴性 (False Negatives): {len(errors['false_negatives'])} 个视频")
        print("(VideoChat2错误地将鬼探头识别为正常交通)")
        for fn in errors['false_negatives']:
            print(f"  {fn['video_id']}: GT={fn['ground_truth']}, 预测={fn['prediction']}")
            print(f"    - sentiment: {fn['sentiment']}, scene_theme: {fn['scene_theme']}")
            print(f"    - key_actions: {fn['key_actions']}")
    
    # 正确分类统计
    print(f"\n✅ 正确分类:")
    correct_ghost = len([x for x in results['predictions']['correct'] if x[3] == "True Positive"])
    correct_normal = len([x for x in results['predictions']['correct'] if x[3] == "True Negative"])
    print(f"正确识别的鬼探头: {correct_ghost}")
    print(f"正确识别的正常交通: {correct_normal}")
    
    print("=" * 80)

def main():
    """主评估函数"""
    
    # 文件路径
    ground_truth_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv"
    videochat2_results_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/videochat/blue_jewel_results/artifacts/outputs"
    
    print("正在加载ground truth标签...")
    ground_truth = load_ground_truth_labels(ground_truth_path)
    print(f"加载了 {len(ground_truth)} 个ground truth标签")
    
    print("正在提取VideoChat2预测结果...")
    predictions = extract_videochat2_predictions(videochat2_results_dir)
    print(f"提取了 {len(predictions)} 个VideoChat2预测")
    
    print("正在计算性能指标...")
    results = calculate_accurate_metrics(ground_truth, predictions)
    
    print("生成详细报告...")
    print_detailed_report(results, ground_truth, predictions)
    
    # 保存详细结果到JSON
    output_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/videochat/accurate_evaluation_results.json"
    
    # 准备保存的数据（去除不能序列化的部分）
    save_data = {
        'metrics': results['metrics'],
        'confusion_matrix': results['confusion_matrix'],
        'error_analysis': results['error_analysis'],
        'summary': {
            'total_videos_evaluated': len(results['common_ids']),
            'ground_truth_distribution': {
                'ghost_probing': sum(1 for label in ground_truth.values() if label == 'ghost_probing'),
                'normal': sum(1 for label in ground_truth.values() if label == 'normal')
            },
            'videochat2_distribution': {
                'ghost_probing': sum(1 for pred_info in predictions.values() if pred_info['prediction'] == 'ghost_probing'),
                'normal': sum(1 for pred_info in predictions.values() if pred_info['prediction'] == 'normal')
            }
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()