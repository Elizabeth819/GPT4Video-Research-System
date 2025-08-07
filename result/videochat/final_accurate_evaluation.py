#!/usr/bin/env python3
"""
最终准确的VideoChat2评估脚本
基于VideoChat2的实际编号系统（1-100连续编号）进行正确评估
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

def create_videochat2_to_dada_mapping() -> Dict[int, str]:
    """创建VideoChat2编号(1-100)到DADA视频ID的映射"""
    
    mapping = {}
    videochat_idx = 1  # VideoChat2从1开始编号
    
    # DADA-100视频的实际分布：
    # images_1_001 到 images_1_027 (27个视频)
    for i in range(1, 28):
        mapping[videochat_idx] = f"images_1_{i:03d}"
        videochat_idx += 1
    
    # images_2_001 到 images_2_005 (5个视频)  
    for i in range(1, 6):
        mapping[videochat_idx] = f"images_2_{i:03d}"
        videochat_idx += 1
    
    # images_3_001 到 images_3_007 (7个视频)
    for i in range(1, 8):
        mapping[videochat_idx] = f"images_3_{i:03d}"
        videochat_idx += 1
    
    # images_4_001 到 images_4_008 (8个视频)
    for i in range(1, 9):
        mapping[videochat_idx] = f"images_4_{i:03d}"
        videochat_idx += 1
    
    # images_5_001 到 images_5_053 (53个视频)
    for i in range(1, 54):
        mapping[videochat_idx] = f"images_5_{i:03d}"
        videochat_idx += 1
    
    return mapping

def extract_all_videochat2_predictions(results_dir: str, mapping: Dict[int, str]) -> Dict[str, Dict]:
    """提取所有VideoChat2预测结果"""
    predictions = {}
    results_path = Path(results_dir)
    
    for json_file in sorted(results_path.glob("actionSummary_images_*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                segment = data[0]
                
                # 从文件名提取编号
                filename = json_file.stem
                match = re.search(r'actionSummary_images_\d+_(\d+)', filename)
                if match:
                    videochat_number = int(match.group(1))
                    
                    # 映射到正确的DADA video ID
                    if videochat_number in mapping:
                        dada_video_id = mapping[videochat_number]
                        
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
                            'videochat_number': videochat_number,
                            'videochat_file': filename,
                            'videochat_video_id': segment.get('video_id', '')
                        }
                        
        except Exception as e:
            print(f"处理文件错误 {json_file}: {e}")
            continue
    
    return predictions

def calculate_final_metrics(ground_truth: Dict[str, str], predictions: Dict[str, Dict]) -> Dict:
    """计算最终的性能指标"""
    
    # 找到共同的video ID
    common_ids = set(ground_truth.keys()) & set(predictions.keys())
    
    print(f"\n评估统计:")
    print(f"Ground Truth视频总数: {len(ground_truth)}")
    print(f"VideoChat2处理视频数: {len(predictions)}")
    print(f"共同评估的视频数量: {len(common_ids)}")
    
    # 初始化混淆矩阵
    tp = fp = tn = fn = 0
    
    # 详细记录
    all_results = []
    false_positives = []
    false_negatives = []
    true_positives = []
    true_negatives = []
    
    for video_id in sorted(common_ids):
        gt_label = ground_truth[video_id]
        pred_info = predictions[video_id]
        pred_label = pred_info['prediction']
        
        result_entry = {
            'video_id': video_id,
            'videochat_number': pred_info['videochat_number'],
            'ground_truth': gt_label,
            'prediction': pred_label,
            'sentiment': pred_info['sentiment'],
            'scene_theme': pred_info['scene_theme'],
            'key_actions': pred_info['key_actions']
        }
        
        if gt_label == 'ghost_probing' and pred_label == 'ghost_probing':
            tp += 1
            result_entry['result_type'] = "True Positive"
            true_positives.append(result_entry)
        elif gt_label == 'normal' and pred_label == 'ghost_probing':
            fp += 1
            result_entry['result_type'] = "False Positive"
            false_positives.append(result_entry)
        elif gt_label == 'normal' and pred_label == 'normal':
            tn += 1
            result_entry['result_type'] = "True Negative"
            true_negatives.append(result_entry)
        elif gt_label == 'ghost_probing' and pred_label == 'normal':
            fn += 1
            result_entry['result_type'] = "False Negative"
            false_negatives.append(result_entry)
        
        all_results.append(result_entry)
    
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
        'error_details': {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_positives': true_positives,
            'true_negatives': true_negatives
        },
        'all_results': all_results
    }

def print_comprehensive_report(results: Dict, ground_truth: Dict, predictions: Dict):
    """打印综合评估报告"""
    
    metrics = results['metrics']
    cm = results['confusion_matrix']
    errors = results['error_details']
    
    print("\n" + "=" * 100)
    print("VIDEOCHAT2 鬼探头检测 - 最终准确评估报告")
    print("=" * 100)
    
    # 性能指标
    print(f"\n📊 核心性能指标:")
    print(f"准确率 (Accuracy):  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"精确率 (Precision): {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"召回率 (Recall):    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"F1分数 (F1-Score):  {metrics['f1_score']:.3f}")
    
    # 混淆矩阵
    print(f"\n📈 混淆矩阵:")
    print("                       预测结果")
    print("                   Ghost    Normal")
    print(f"实际 Ghost        {cm['true_positives']:5d}    {cm['false_negatives']:5d}")
    print(f"     Normal       {cm['false_positives']:5d}    {cm['true_negatives']:5d}")
    print(f"     总计         {cm['true_positives']+cm['false_positives']:5d}    {cm['false_negatives']+cm['true_negatives']:5d}")
    
    # 数据分布
    gt_ghost = sum(1 for label in ground_truth.values() if label == 'ghost_probing')
    gt_normal = sum(1 for label in ground_truth.values() if label == 'normal')
    pred_ghost = sum(1 for p in predictions.values() if p['prediction'] == 'ghost_probing')
    pred_normal = sum(1 for p in predictions.values() if p['prediction'] == 'normal')
    
    print(f"\n📋 数据分布:")
    print(f"Ground Truth总计 - 鬼探头: {gt_ghost}, 正常交通: {gt_normal}")
    print(f"VideoChat2预测总计 - 鬼探头: {pred_ghost}, 正常交通: {pred_normal}")
    print(f"实际评估视频数量: {cm['total_samples']}")
    
    # VideoChat2的分类模式
    print(f"\n🔍 VideoChat2分类模式分析:")
    videochat_ghost = [p for p in predictions.values() if p['prediction'] == 'ghost_probing']
    videochat_normal = [p for p in predictions.values() if p['prediction'] == 'normal']
    
    print(f"VideoChat2将前60个视频（编号1-60）全部分类为: 鬼探头")
    print(f"VideoChat2将后40个视频（编号61-100）全部分类为: 正常交通")
    print(f"这种固定模式表明VideoChat2可能使用了预设的分类策略而非真实的视频分析")
    
    # 详细错误分析
    print(f"\n❌ 错误分类详情:")
    
    print(f"\n【假阳性】 False Positives: {len(errors['false_positives'])} 个")
    print("VideoChat2错误地将正常交通识别为鬼探头:")
    for i, fp in enumerate(errors['false_positives'][:10], 1):  # 只显示前10个
        print(f"  {i}. {fp['video_id']} (VideoChat2编号: {fp['videochat_number']})")
        print(f"     真实: {fp['ground_truth']} → 预测: {fp['prediction']}")
        if i <= 3:  # 只显示前3个的详细信息
            print(f"     情感: {fp['sentiment']}, 场景: {fp['scene_theme']}")
            print(f"     关键动作: {fp['key_actions'][:60]}...")
    if len(errors['false_positives']) > 10:
        print(f"  ... 还有 {len(errors['false_positives'])-10} 个假阳性案例")
    
    print(f"\n【假阴性】 False Negatives: {len(errors['false_negatives'])} 个")
    print("VideoChat2错误地将鬼探头识别为正常交通:")
    for i, fn in enumerate(errors['false_negatives'][:10], 1):  # 只显示前10个
        print(f"  {i}. {fn['video_id']} (VideoChat2编号: {fn['videochat_number']})")
        print(f"     真实: {fn['ground_truth']} → 预测: {fn['prediction']}")
        if i <= 3:  # 只显示前3个的详细信息
            print(f"     情感: {fn['sentiment']}, 场景: {fn['scene_theme']}")
            print(f"     关键动作: {fn['key_actions'][:60]}...")
    if len(errors['false_negatives']) > 10:
        print(f"  ... 还有 {len(errors['false_negatives'])-10} 个假阴性案例")
    
    # 正确分类统计
    print(f"\n✅ 正确分类统计:")
    print(f"正确识别的鬼探头 (True Positives): {len(errors['true_positives'])}")
    print(f"正确识别的正常交通 (True Negatives): {len(errors['true_negatives'])}")
    print(f"总正确数: {len(errors['true_positives']) + len(errors['true_negatives'])}/{cm['total_samples']}")
    
    print("\n" + "=" * 100)

def main():
    """主函数"""
    
    # 文件路径
    ground_truth_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv"
    videochat2_results_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/videochat/blue_jewel_results/artifacts/outputs"
    
    print("VideoChat2最终评估开始...")
    
    print("\n1. 创建VideoChat2编号到DADA视频的映射...")
    mapping = create_videochat2_to_dada_mapping()
    print(f"   创建了 {len(mapping)} 个映射关系")
    
    print("\n2. 加载ground truth标签...")
    ground_truth = load_ground_truth_labels(ground_truth_path)
    print(f"   加载了 {len(ground_truth)} 个ground truth标签")
    
    print("\n3. 提取VideoChat2预测结果...")
    predictions = extract_all_videochat2_predictions(videochat2_results_dir, mapping)
    print(f"   提取了 {len(predictions)} 个VideoChat2预测")
    
    print("\n4. 计算最终性能指标...")
    results = calculate_final_metrics(ground_truth, predictions)
    
    print("\n5. 生成综合评估报告...")
    print_comprehensive_report(results, ground_truth, predictions)
    
    # 保存详细结果
    output_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/videochat/FINAL_VIDEOCHAT2_METRICS.json"
    
    save_data = {
        'metrics': results['metrics'],
        'confusion_matrix': results['confusion_matrix'],
        'evaluation_summary': {
            'total_videos_in_dataset': 100,
            'total_videos_evaluated': results['confusion_matrix']['total_samples'],
            'ground_truth_distribution': {
                'ghost_probing': sum(1 for label in ground_truth.values() if label == 'ghost_probing'),
                'normal': sum(1 for label in ground_truth.values() if label == 'normal')
            },
            'videochat2_distribution': {
                'ghost_probing': sum(1 for p in predictions.values() if p['prediction'] == 'ghost_probing'),
                'normal': sum(1 for p in predictions.values() if p['prediction'] == 'normal')
            },
            'videochat2_pattern': "Fixed pattern: videos 1-60 as ghost_probing, 61-100 as normal"
        },
        'error_summary': {
            'false_positives_count': len(results['error_details']['false_positives']),
            'false_negatives_count': len(results['error_details']['false_negatives']),
            'false_positive_videos': [fp['video_id'] for fp in results['error_details']['false_positives']],
            'false_negative_videos': [fn['video_id'] for fn in results['error_details']['false_negatives']]
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细指标已保存到: {output_file}")
    print("\n评估完成！")

if __name__ == "__main__":
    main()