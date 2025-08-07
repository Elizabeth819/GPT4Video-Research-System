#!/usr/bin/env python3
"""
评估改进版LLaVA检测结果
对比ground truth计算性能指标
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def load_ground_truth(csv_path: str) -> Dict[str, bool]:
    """加载ground truth标签"""
    try:
        df = pd.read_csv(csv_path, sep='\t')
        print(f"📊 读取ground truth文件: {csv_path}")
        
        gt_labels = {}
        for _, row in df.iterrows():
            video_id = str(row['video_id']).replace('.avi', '')
            label = str(row['ground_truth_label']).lower()
            
            # 判断是否为鬼探头
            has_ghost_probing = (
                'ghost probing' in label or 
                'ghost' in label or
                ('s:' in label and 'none' not in label and 'cut-in' not in label)
            )
            gt_labels[video_id] = has_ghost_probing
            
        print(f"📊 加载了 {len(gt_labels)} 个ground truth标签")
        return gt_labels
        
    except Exception as e:
        print(f"❌ 加载ground truth失败: {e}")
        return {}

def load_improved_results(json_path: str) -> List[Dict]:
    """加载改进版检测结果"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        print(f"📊 加载了 {len(results)} 个改进版检测结果")
        
        return results
        
    except Exception as e:
        print(f"❌ 加载检测结果失败: {e}")
        return []

def compare_with_baseline(improved_results: List[Dict], baseline_results: List[Dict]) -> Dict:
    """对比改进版与基线版本的结果"""
    
    # 创建视频ID到结果的映射
    improved_map = {r['video_id']: r for r in improved_results}
    baseline_map = {r['video_id']: r for r in baseline_results}
    
    comparison = {
        'total_videos': len(improved_results),
        'changes': [],
        'detection_changes': {
            'baseline_detected': 0,
            'improved_detected': 0,
            'newly_detected': [],
            'no_longer_detected': []
        }
    }
    
    for video_id in improved_map:
        if video_id in baseline_map:
            improved = improved_map[video_id]
            baseline = baseline_map[video_id]
            
            improved_detected = improved.get('ghost_probing_label') == 'yes'
            baseline_detected = baseline.get('ghost_probing_label') == 'yes'
            
            if baseline_detected:
                comparison['detection_changes']['baseline_detected'] += 1
            if improved_detected:
                comparison['detection_changes']['improved_detected'] += 1
            
            # 检测变化
            if improved_detected and not baseline_detected:
                comparison['detection_changes']['newly_detected'].append({
                    'video_id': video_id,
                    'improved_confidence': improved.get('confidence', 0),
                    'baseline_confidence': baseline.get('confidence', 0)
                })
            elif baseline_detected and not improved_detected:
                comparison['detection_changes']['no_longer_detected'].append({
                    'video_id': video_id,
                    'improved_confidence': improved.get('confidence', 0),
                    'baseline_confidence': baseline.get('confidence', 0)
                })
            
            # 记录详细变化
            comparison['changes'].append({
                'video_id': video_id,
                'baseline_detected': baseline_detected,
                'improved_detected': improved_detected,
                'baseline_confidence': baseline.get('confidence', 0),
                'improved_confidence': improved.get('confidence', 0),
                'confidence_change': improved.get('confidence', 0) - baseline.get('confidence', 0),
                'max_frame_change': improved.get('temporal_analysis', {}).get('max_change', 0)
            })
    
    return comparison

def evaluate_improved_performance(gt_labels: Dict[str, bool], detection_results: List[Dict]) -> Dict:
    """评估改进版检测性能"""
    
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
            
            # 时序分析数据
            temporal_analysis = result.get('temporal_analysis', {})
            detection_scores = result.get('detection_scores', {})
            
            matched_results.append({
                'video_id': video_id,
                'ground_truth': ground_truth,
                'detected': detected,
                'confidence': confidence,
                'max_frame_change': temporal_analysis.get('max_change', 0),
                'avg_frame_change': temporal_analysis.get('mean_change', 0),
                'sudden_changes': temporal_analysis.get('sudden_change_count', 0),
                'continuous_regions': len(temporal_analysis.get('continuous_change_regions', [])),
                'detection_scores': detection_scores,
                'thresholds_used': result.get('thresholds_used', {})
            })
    
    print(f"📊 匹配到 {len(matched_results)} 个有ground truth的检测结果")
    
    if not matched_results:
        return {"error": "没有匹配的结果"}
    
    # 计算混淆矩阵
    tp = sum(1 for r in matched_results if r['ground_truth'] and r['detected'])
    tn = sum(1 for r in matched_results if not r['ground_truth'] and not r['detected'])
    fp = sum(1 for r in matched_results if not r['ground_truth'] and r['detected'])
    fn = sum(1 for r in matched_results if r['ground_truth'] and not r['detected'])
    
    total = len(matched_results)
    
    # 计算性能指标
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 分析检测案例
    true_positives = [r for r in matched_results if r['ground_truth'] and r['detected']]
    false_positives = [r for r in matched_results if not r['ground_truth'] and r['detected']]
    false_negatives = [r for r in matched_results if r['ground_truth'] and not r['detected']]
    true_negatives = [r for r in matched_results if not r['ground_truth'] and not r['detected']]
    
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
        'detailed_analysis': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives
        },
        'threshold_analysis': {
            'avg_tp_confidence': np.mean([r['confidence'] for r in true_positives]) if true_positives else 0,
            'avg_fp_confidence': np.mean([r['confidence'] for r in false_positives]) if false_positives else 0,
            'avg_tp_max_change': np.mean([r['max_frame_change'] for r in true_positives]) if true_positives else 0,
            'avg_fp_max_change': np.mean([r['max_frame_change'] for r in false_positives]) if false_positives else 0,
        }
    }
    
    return results

def print_improved_evaluation_report(eval_results: Dict, comparison: Dict = None):
    """打印改进版评估报告"""
    
    print("\n" + "="*80)
    print("🚀 改进版LLaVA鬼探头检测性能评估报告")
    print("="*80)
    
    if 'error' in eval_results:
        print(f"❌ 评估错误: {eval_results['error']}")
        return
    
    # 基本信息
    total = eval_results['total_videos']
    cm = eval_results['confusion_matrix']
    metrics = eval_results['performance_metrics']
    threshold_analysis = eval_results['threshold_analysis']
    
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
    
    # 阈值分析
    print("🔍 检测置信度分析:")
    print(f"   真阳性平均置信度: {threshold_analysis['avg_tp_confidence']:.3f}")
    print(f"   假阳性平均置信度: {threshold_analysis['avg_fp_confidence']:.3f}")
    print(f"   真阳性平均帧变化: {threshold_analysis['avg_tp_max_change']:.4f}")
    print(f"   假阳性平均帧变化: {threshold_analysis['avg_fp_max_change']:.4f}")
    print()
    
    # 详细案例分析
    detailed = eval_results['detailed_analysis']
    
    if detailed['true_positives']:
        print("✅ 正确检测出的鬼探头:")
        for tp in detailed['true_positives']:
            print(f"   - {tp['video_id']}: 置信度={tp['confidence']:.3f}, 最大变化={tp['max_frame_change']:.4f}")
    
    if detailed['false_positives']:
        print("\n❌ 误报的正常视频:")
        for fp in detailed['false_positives']:
            print(f"   - {fp['video_id']}: 置信度={fp['confidence']:.3f}, 最大变化={fp['max_frame_change']:.4f}")
    
    if detailed['false_negatives']:
        print("\n⚠️  漏检的鬼探头:")
        for fn in detailed['false_negatives']:
            print(f"   - {fn['video_id']}: 置信度={fn['confidence']:.3f}, 最大变化={fn['max_frame_change']:.4f}")
    print()
    
    # 对比分析
    if comparison:
        print("📈 改进效果分析:")
        baseline_detected = comparison['detection_changes']['baseline_detected']
        improved_detected = comparison['detection_changes']['improved_detected']
        newly_detected = len(comparison['detection_changes']['newly_detected'])
        no_longer_detected = len(comparison['detection_changes']['no_longer_detected'])
        
        print(f"   基线版本检测数: {baseline_detected}")
        print(f"   改进版本检测数: {improved_detected}")
        print(f"   新增检测: {newly_detected} 个")
        print(f"   不再检测: {no_longer_detected} 个")
        
        if comparison['detection_changes']['newly_detected']:
            print("   新增检测的视频:")
            for item in comparison['detection_changes']['newly_detected']:
                print(f"     - {item['video_id']}: 置信度 {item['baseline_confidence']:.3f} → {item['improved_confidence']:.3f}")
    
    print("="*80)

def main():
    """主函数"""
    print("🚀 开始评估改进版LLaVA检测结果...")
    
    # 文件路径
    gt_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv"
    improved_results_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/minimal_job/improved_results/artifacts/outputs/results/improved_llava_results_20250722_025127.json"
    baseline_results_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/minimal_job/strict_validation_results/artifacts/outputs/results/strict_llava_results_20250722_021252.json"
    
    # 检查文件存在性
    if not Path(gt_file).exists():
        print(f"❌ Ground truth文件不存在: {gt_file}")
        return
    
    if not Path(improved_results_file).exists():
        print(f"❌ 改进版结果文件不存在: {improved_results_file}")
        return
    
    # 加载数据
    gt_labels = load_ground_truth(gt_file)
    if not gt_labels:
        print("❌ 无法加载ground truth标签")
        return
    
    improved_results = load_improved_results(improved_results_file)
    if not improved_results:
        print("❌ 无法加载改进版检测结果")
        return
    
    # 加载基线结果进行对比
    comparison = None
    if Path(baseline_results_file).exists():
        try:
            with open(baseline_results_file, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
            baseline_results = baseline_data.get('results', [])
            if baseline_results:
                comparison = compare_with_baseline(improved_results, baseline_results)
                print(f"📊 加载了基线结果进行对比: {len(baseline_results)} 个")
        except Exception as e:
            print(f"⚠️  无法加载基线结果: {e}")
    
    # 评估性能
    eval_results = evaluate_improved_performance(gt_labels, improved_results)
    
    # 打印报告
    print_improved_evaluation_report(eval_results, comparison)
    
    # 保存详细结果
    output_file = "improved_evaluation_report.json"
    report_data = {
        'evaluation_results': eval_results,
        'comparison_with_baseline': comparison,
        'timestamp': '20250722_025127'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 详细评估结果已保存到: {output_file}")

if __name__ == "__main__":
    main()