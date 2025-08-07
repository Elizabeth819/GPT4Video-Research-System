#!/usr/bin/env python3
"""
评估严格验证LLaVA检测结果
计算准确率、精确度、召回率等指标
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def load_ground_truth(csv_path: str) -> Dict[str, bool]:
    """加载ground truth标签"""
    try:
        df = pd.read_csv(csv_path, sep='\t')  # 使用tab分隔符
        print(f"📊 读取ground truth文件: {csv_path}")
        print(f"📊 列名: {df.columns.tolist()}")
        
        gt_labels = {}
        for _, row in df.iterrows():
            video_id = str(row['video_id']).replace('.avi', '')  # 移除后缀统一格式
            label = str(row['ground_truth_label']).lower()
            
            # 判断是否为鬼探头
            has_ghost_probing = (
                'ghost probing' in label or 
                'ghost' in label or
                ('s:' in label and 'none' not in label and 'cut-in' not in label)
            )
            gt_labels[video_id] = has_ghost_probing
            
        print(f"📊 加载了 {len(gt_labels)} 个ground truth标签")
        
        # 统计ground truth分布
        ghost_count = sum(gt_labels.values())
        normal_count = len(gt_labels) - ghost_count
        print(f"📊 Ground Truth分布:")
        print(f"   - 鬼探头视频: {ghost_count}")
        print(f"   - 正常视频: {normal_count}")
        print(f"   - 鬼探头比例: {ghost_count/len(gt_labels)*100:.1f}%")
        
        return gt_labels
        
    except Exception as e:
        print(f"❌ 加载ground truth失败: {e}")
        return {}

def load_strict_results(json_path: str) -> List[Dict]:
    """加载严格验证检测结果"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        print(f"📊 加载了 {len(results)} 个检测结果")
        
        return results
        
    except Exception as e:
        print(f"❌ 加载检测结果失败: {e}")
        return []

def evaluate_performance(gt_labels: Dict[str, bool], detection_results: List[Dict]) -> Dict:
    """评估检测性能"""
    
    # 准备评估数据
    matched_results = []
    
    for result in detection_results:
        video_id = result.get('video_id', '').replace('.avi', '')
        
        if video_id in gt_labels:
            # 检测结果
            detected = result.get('ghost_probing_label', 'no').lower() == 'yes'
            confidence = result.get('confidence', 0.0)
            
            # Ground truth
            ground_truth = gt_labels[video_id]
            
            matched_results.append({
                'video_id': video_id,
                'ground_truth': ground_truth,
                'detected': detected,
                'confidence': confidence,
                'max_frame_change': result.get('max_frame_change', 0),
                'avg_frame_change': result.get('avg_frame_change', 0)
            })
    
    print(f"📊 匹配到 {len(matched_results)} 个有ground truth的检测结果")
    
    if not matched_results:
        return {"error": "没有匹配的结果"}
    
    # 计算混淆矩阵
    tp = sum(1 for r in matched_results if r['ground_truth'] and r['detected'])      # True Positive
    tn = sum(1 for r in matched_results if not r['ground_truth'] and not r['detected'])  # True Negative  
    fp = sum(1 for r in matched_results if not r['ground_truth'] and r['detected'])      # False Positive
    fn = sum(1 for r in matched_results if r['ground_truth'] and not r['detected'])      # False Negative
    
    total = len(matched_results)
    
    # 计算性能指标
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 统计特征变化
    ghost_videos = [r for r in matched_results if r['ground_truth']]
    normal_videos = [r for r in matched_results if not r['ground_truth']]
    
    ghost_max_changes = [r['max_frame_change'] for r in ghost_videos]
    normal_max_changes = [r['max_frame_change'] for r in normal_videos]
    
    results = {
        'total_videos': total,
        'confusion_matrix': {
            'true_positive': tp,
            'true_negative': tn, 
            'false_positive': fp,
            'false_negative': fn
        },
        'performance_metrics': {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'specificity': round(specificity, 4),
            'f1_score': round(f1_score, 4)
        },
        'feature_analysis': {
            'ghost_videos_max_change': {
                'mean': round(np.mean(ghost_max_changes), 4) if ghost_max_changes else 0,
                'std': round(np.std(ghost_max_changes), 4) if ghost_max_changes else 0,
                'max': round(max(ghost_max_changes), 4) if ghost_max_changes else 0,
                'min': round(min(ghost_max_changes), 4) if ghost_max_changes else 0
            },
            'normal_videos_max_change': {
                'mean': round(np.mean(normal_max_changes), 4) if normal_max_changes else 0,
                'std': round(np.std(normal_max_changes), 4) if normal_max_changes else 0,
                'max': round(max(normal_max_changes), 4) if normal_max_changes else 0,
                'min': round(min(normal_max_changes), 4) if normal_max_changes else 0
            }
        },
        'detailed_results': matched_results
    }
    
    return results

def print_evaluation_report(eval_results: Dict):
    """打印评估报告"""
    
    print("\n" + "="*80)
    print("🎯 严格验证LLaVA鬼探头检测性能评估报告")
    print("="*80)
    
    if 'error' in eval_results:
        print(f"❌ 评估错误: {eval_results['error']}")
        return
    
    # 基本信息
    total = eval_results['total_videos']
    cm = eval_results['confusion_matrix']
    metrics = eval_results['performance_metrics']
    
    print(f"📊 评估视频总数: {total}")
    print()
    
    # 混淆矩阵
    print("📋 混淆矩阵:")
    print(f"   真阳性 (TP): {cm['true_positive']:3d}  |  假阳性 (FP): {cm['false_positive']:3d}")
    print(f"   假阴性 (FN): {cm['false_negative']:3d}  |  真阴性 (TN): {cm['true_negative']:3d}")
    print()
    
    # 性能指标
    print("🎯 性能指标:")
    print(f"   准确率 (Accuracy):  {metrics['accuracy']:.1%}")
    print(f"   精确度 (Precision): {metrics['precision']:.1%}")
    print(f"   召回率 (Recall):    {metrics['recall']:.1%}")
    print(f"   特异性 (Specificity): {metrics['specificity']:.1%}")
    print(f"   F1分数 (F1-Score):  {metrics['f1_score']:.1%}")
    print()
    
    # 特征分析
    feature_analysis = eval_results['feature_analysis']
    ghost_stats = feature_analysis['ghost_videos_max_change']
    normal_stats = feature_analysis['normal_videos_max_change']
    
    print("🔍 特征变化分析:")
    print(f"   鬼探头视频最大变化: 均值={ghost_stats['mean']:.4f}, 标准差={ghost_stats['std']:.4f}")
    print(f"                     最大值={ghost_stats['max']:.4f}, 最小值={ghost_stats['min']:.4f}")
    print(f"   正常视频最大变化:   均值={normal_stats['mean']:.4f}, 标准差={normal_stats['std']:.4f}")
    print(f"                     最大值={normal_stats['max']:.4f}, 最小值={normal_stats['min']:.4f}")
    print()
    
    # 问题分析
    print("⚠️  问题分析:")
    if metrics['recall'] < 0.1:
        print("   - 召回率极低，几乎无法检测到真实的鬼探头")
    if metrics['precision'] < 0.1:
        print("   - 精确度极低，存在大量误报")
    if ghost_stats['max'] < 0.3:
        print("   - 鬼探头视频的特征变化仍然很小，可能需要更敏感的检测方法")
    if normal_stats['max'] > 0.2:
        print("   - 正常视频也有较大的特征变化，可能影响判断阈值设置")
    
    print("="*80)

def main():
    """主函数"""
    print("🎯 开始评估严格验证LLaVA检测结果...")
    
    # 文件路径
    gt_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv"
    strict_results_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/minimal_job/strict_validation_results/artifacts/outputs/results/strict_llava_results_20250722_021252.json"
    
    # 检查文件存在性
    if not Path(gt_file).exists():
        print(f"❌ Ground truth文件不存在: {gt_file}")
        return
    
    if not Path(strict_results_file).exists():
        print(f"❌ 检测结果文件不存在: {strict_results_file}")
        return
    
    # 加载数据
    gt_labels = load_ground_truth(gt_file)
    if not gt_labels:
        print("❌ 无法加载ground truth标签")
        return
    
    detection_results = load_strict_results(strict_results_file)
    if not detection_results:
        print("❌ 无法加载检测结果")
        return
    
    # 评估性能
    eval_results = evaluate_performance(gt_labels, detection_results)
    
    # 打印报告
    print_evaluation_report(eval_results)
    
    # 保存详细结果
    output_file = "strict_validation_evaluation_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"📄 详细评估结果已保存到: {output_file}")

if __name__ == "__main__":
    main()