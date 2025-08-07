#!/usr/bin/env python3
"""
修正的VideoChat2评估脚本
解决VideoChat2结果文件编号与ground truth编号不匹配的问题
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
            if not row['video_id']:
                continue
                
            video_id = row['video_id'].replace('.avi', '')
            label = row['ground_truth_label']
            
            # 转换为二分类
            if label == 'none':
                ground_truth[video_id] = 'normal'
            elif 'ghost probing' in label:
                ground_truth[video_id] = 'ghost_probing'
            elif label == 'cut-in' or 'cut-in' in label:
                ground_truth[video_id] = 'normal'
    
    return ground_truth

def create_video_mapping() -> Dict[str, str]:
    """创建VideoChat2结果文件到DADA视频的正确映射"""
    
    # VideoChat2处理了100个视频，但使用了连续编号021-120
    # 需要映射到正确的DADA video IDs
    
    mapping = {}
    
    # DADA-100视频的实际分布：
    # images_1_001 到 images_1_027 (27个视频)
    # images_2_001 到 images_2_005 (5个视频)  
    # images_3_001 到 images_3_007 (7个视频)
    # images_4_001 到 images_4_008 (8个视频)
    # images_5_001 到 images_5_053 (53个视频)
    
    # VideoChat2的编号从021开始，映射到实际视频
    videochat_idx = 21  # 从021开始
    
    # Category 1: images_1_001 到 images_1_027
    for i in range(1, 28):
        vc_id = f"images_1_{videochat_idx:03d}"
        dada_id = f"images_1_{i:03d}"
        mapping[vc_id] = dada_id
        videochat_idx += 1
    
    # Category 2: images_2_001 到 images_2_005
    for i in range(1, 6):
        vc_id = f"images_2_{videochat_idx:03d}"
        dada_id = f"images_2_{i:03d}"
        mapping[vc_id] = dada_id
        videochat_idx += 1
    
    # Category 3: images_3_001 到 images_3_007
    for i in range(1, 8):
        vc_id = f"images_3_{videochat_idx:03d}"
        dada_id = f"images_3_{i:03d}"
        mapping[vc_id] = dada_id
        videochat_idx += 1
    
    # Category 4: images_4_001 到 images_4_008
    for i in range(1, 9):
        vc_id = f"images_4_{videochat_idx:03d}"
        dada_id = f"images_4_{i:03d}"
        mapping[vc_id] = dada_id
        videochat_idx += 1
    
    # Category 5: images_5_001 到 images_5_053
    for i in range(1, 54):
        vc_id = f"images_5_{videochat_idx:03d}"
        dada_id = f"images_5_{i:03d}"
        mapping[vc_id] = dada_id
        videochat_idx += 1
    
    return mapping

def extract_videochat2_predictions_with_mapping(results_dir: str, mapping: Dict[str, str]) -> Dict[str, Dict]:
    """使用正确映射提取VideoChat2预测结果"""
    predictions = {}
    results_path = Path(results_dir)
    
    for json_file in results_path.glob("actionSummary_images_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                segment = data[0]
                
                # 从文件名提取VideoChat2的ID
                filename = json_file.stem
                match = re.search(r'actionSummary_images_(\d+)_(\d+)', filename)
                if match:
                    category, number = match.groups()
                    vc_video_id = f"images_{category}_{number}"
                    
                    # 映射到正确的DADA video ID
                    if vc_video_id in mapping:
                        dada_video_id = mapping[vc_video_id]
                        
                        # 获取VideoChat2的分类信息
                        sentiment = segment.get('sentiment', '')
                        scene_theme = segment.get('scene_theme', '')
                        key_actions = segment.get('key_actions', '')
                        summary = segment.get('summary', '')
                        
                        # VideoChat2的分类逻辑
                        if (sentiment == 'Negative' and 
                            scene_theme == 'Dramatic' and 
                            'ghost probing' in key_actions.lower()):
                            prediction = 'ghost_probing'
                        else:
                            prediction = 'normal'
                        
                        predictions[dada_video_id] = {
                            'prediction': prediction,
                            'sentiment': sentiment,
                            'scene_theme': scene_theme,
                            'key_actions': key_actions,
                            'summary': summary[:100] + '...' if len(summary) > 100 else summary,
                            'videochat_id': vc_video_id,
                            'file': str(json_file)
                        }
                        
        except Exception as e:
            print(f"处理文件错误 {json_file}: {e}")
            continue
    
    return predictions

def calculate_corrected_metrics(ground_truth: Dict[str, str], predictions: Dict[str, Dict]) -> Dict:
    """计算修正后的性能指标"""
    
    # 找到共同的video ID
    common_ids = set(ground_truth.keys()) & set(predictions.keys())
    
    print(f"Ground Truth视频数量: {len(ground_truth)}")
    print(f"VideoChat2预测数量: {len(predictions)}")
    print(f"共同评估的视频数量: {len(common_ids)}")
    
    if not common_ids:
        raise ValueError("没有找到共同的video ID")
    
    # 初始化混淆矩阵
    tp = fp = tn = fn = 0
    
    # 详细记录
    all_results = []
    false_positives = []
    false_negatives = []
    
    for video_id in sorted(common_ids):
        gt_label = ground_truth[video_id]
        pred_info = predictions[video_id]
        pred_label = pred_info['prediction']
        
        result_type = ""
        is_correct = False
        
        if gt_label == 'ghost_probing' and pred_label == 'ghost_probing':
            tp += 1
            result_type = "True Positive"
            is_correct = True
        elif gt_label == 'normal' and pred_label == 'ghost_probing':
            fp += 1
            result_type = "False Positive"
            false_positives.append({
                'video_id': video_id,
                'videochat_id': pred_info['videochat_id'],
                'ground_truth': gt_label,
                'prediction': pred_label,
                'key_actions': pred_info['key_actions'],
                'sentiment': pred_info['sentiment'],
                'scene_theme': pred_info['scene_theme']
            })
        elif gt_label == 'normal' and pred_label == 'normal':
            tn += 1
            result_type = "True Negative"
            is_correct = True
        elif gt_label == 'ghost_probing' and pred_label == 'normal':
            fn += 1
            result_type = "False Negative"
            false_negatives.append({
                'video_id': video_id,
                'videochat_id': pred_info['videochat_id'],
                'ground_truth': gt_label,
                'prediction': pred_label,
                'key_actions': pred_info['key_actions'],
                'sentiment': pred_info['sentiment'],
                'scene_theme': pred_info['scene_theme']
            })
        
        all_results.append({
            'video_id': video_id,
            'videochat_id': pred_info['videochat_id'],
            'ground_truth': gt_label,
            'prediction': pred_label,
            'result_type': result_type,
            'is_correct': is_correct,
            'sentiment': pred_info['sentiment'],
            'scene_theme': pred_info['scene_theme'],
            'key_actions': pred_info['key_actions'][:50] + '...' if len(pred_info['key_actions']) > 50 else pred_info['key_actions']
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
        'error_analysis': {
            'false_positives': false_positives,
            'false_negatives': false_negatives
        },
        'all_results': all_results,
        'common_ids': sorted(list(common_ids))
    }

def print_final_report(results: Dict, ground_truth: Dict, predictions: Dict):
    """打印最终评估报告"""
    
    metrics = results['metrics']
    cm = results['confusion_matrix']
    errors = results['error_analysis']
    
    print("=" * 90)
    print("VIDEOCHAT2 最终准确评估报告")
    print("=" * 90)
    
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
    
    # 数据分布
    gt_ghost = sum(1 for label in ground_truth.values() if label == 'ghost_probing')
    gt_normal = sum(1 for label in ground_truth.values() if label == 'normal')
    pred_ghost = len([r for r in results['all_results'] if r['prediction'] == 'ghost_probing'])
    pred_normal = len([r for r in results['all_results'] if r['prediction'] == 'normal'])
    
    print(f"\n📋 数据分布:")
    print(f"Ground Truth - 鬼探头: {gt_ghost}, 正常交通: {gt_normal}")
    print(f"VideoChat2预测 - 鬼探头: {pred_ghost}, 正常交通: {pred_normal}")
    print(f"共同评估视频数量: {cm['total_samples']}")
    
    # 详细错误分析
    print(f"\n❌ 详细错误分析:")
    
    if errors['false_positives']:
        print(f"\n假阳性 (False Positives): {len(errors['false_positives'])} 个视频")
        print("VideoChat2错误地将正常交通识别为鬼探头:")
        for i, fp in enumerate(errors['false_positives'], 1):
            print(f"  {i}. {fp['video_id']} (VideoChat2文件: {fp['videochat_id']})")
            print(f"     GT: {fp['ground_truth']} → 预测: {fp['prediction']}")
            print(f"     情感: {fp['sentiment']}, 场景: {fp['scene_theme']}")
            print(f"     关键动作: {fp['key_actions']}")
            print()
    
    if errors['false_negatives']:
        print(f"假阴性 (False Negatives): {len(errors['false_negatives'])} 个视频")
        print("VideoChat2错误地将鬼探头识别为正常交通:")
        for i, fn in enumerate(errors['false_negatives'], 1):
            print(f"  {i}. {fn['video_id']} (VideoChat2文件: {fn['videochat_id']})")
            print(f"     GT: {fn['ground_truth']} → 预测: {fn['prediction']}")
            print(f"     情感: {fn['sentiment']}, 场景: {fn['scene_theme']}")
            print(f"     关键动作: {fn['key_actions']}")
            print()
    
    # 正确分类统计
    correct_results = [r for r in results['all_results'] if r['is_correct']]
    correct_ghost = len([r for r in correct_results if r['result_type'] == "True Positive"])
    correct_normal = len([r for r in correct_results if r['result_type'] == "True Negative"])
    
    print(f"✅ 正确分类统计:")
    print(f"正确识别的鬼探头: {correct_ghost}")
    print(f"正确识别的正常交通: {correct_normal}")
    print(f"总正确数: {len(correct_results)}/{cm['total_samples']}")
    
    print("=" * 90)

def main():
    """主函数"""
    
    # 文件路径
    ground_truth_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv"
    videochat2_results_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/videochat/blue_jewel_results/artifacts/outputs"
    
    print("创建VideoChat2结果到DADA视频的映射...")
    mapping = create_video_mapping()
    print(f"创建了 {len(mapping)} 个映射关系")
    
    print("\n加载ground truth标签...")
    ground_truth = load_ground_truth_labels(ground_truth_path)
    print(f"加载了 {len(ground_truth)} 个ground truth标签")
    
    print("\n提取VideoChat2预测结果（使用正确映射）...")
    predictions = extract_videochat2_predictions_with_mapping(videochat2_results_dir, mapping)
    print(f"提取了 {len(predictions)} 个VideoChat2预测")
    
    print("\n计算修正后的性能指标...")
    results = calculate_corrected_metrics(ground_truth, predictions)
    
    print("\n生成最终评估报告...")
    print_final_report(results, ground_truth, predictions)
    
    # 保存详细结果
    output_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/videochat/final_corrected_evaluation.json"
    
    save_data = {
        'metrics': results['metrics'],
        'confusion_matrix': results['confusion_matrix'],
        'error_analysis': results['error_analysis'],
        'all_results': results['all_results'],
        'evaluation_info': {
            'total_videos_evaluated': len(results['common_ids']),
            'mapping_used': True,
            'ground_truth_count': len(ground_truth),
            'predictions_count': len(predictions)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()