#!/usr/bin/env python3
"""
评估GPT-4.1平衡版100视频检测结果
对比ground truth计算准确率、精确度、召回率
输出格式与GPT-4.1保持一致
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

def load_ground_truth_labels(csv_path: str) -> Dict[str, bool]:
    """加载ground truth标签"""
    try:
        df = pd.read_csv(csv_path, sep='\t')
        print(f"📊 读取ground truth文件: {csv_path}")
        print(f"📊 列名: {df.columns.tolist()}")
        
        gt_labels = {}
        ghost_count = 0
        
        for _, row in df.iterrows():
            video_id = str(row['video_id']).replace('.avi', '')
            label = str(row['ground_truth_label']).lower()
            
            # 判断是否为鬼探头 (与GPT-4.1评估标准一致)
            has_ghost_probing = (
                'ghost probing' in label or 
                'ghost' in label or
                ('s:' in label and 'none' not in label and 'cut-in' not in label)
            )
            
            gt_labels[video_id] = has_ghost_probing
            if has_ghost_probing:
                ghost_count += 1
        
        print(f"📊 加载了 {len(gt_labels)} 个ground truth标签")
        print(f"📊 Ground Truth分布:")
        print(f"   - 鬼探头视频: {ghost_count}")
        print(f"   - 正常视频: {len(gt_labels) - ghost_count}")
        print(f"   - 鬼探头比例: {ghost_count/len(gt_labels)*100:.1f}%")
        
        return gt_labels
        
    except Exception as e:
        print(f"❌ 加载ground truth失败: {e}")
        return {}

def load_gpt41_results(json_path: str) -> List[Dict]:
    """加载GPT-4.1检测结果"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        metadata = data.get('metadata', {})
        
        print(f"📊 加载了 {len(results)} 个GPT-4.1检测结果")
        print(f"📊 模型信息: {metadata.get('model', 'Unknown')}")
        print(f"📊 成功处理: {metadata.get('successful_videos', 0)}")
        print(f"📊 处理失败: {metadata.get('failed_videos', 0)}")
        
        return results
        
    except Exception as e:
        print(f"❌ 加载GPT-4.1检测结果失败: {e}")
        return []

def extract_ghost_probing_detection(result: Dict) -> Tuple[bool, bool, float]:
    """从GPT-4.1结果中提取鬼探头检测信息"""
    
    key_actions = result.get('key_actions', '').lower()
    
    # 检测鬼探头关键词 (与GPT-4.1标准一致)
    high_confidence_ghost = 'ghost probing' in key_actions and 'potential' not in key_actions
    potential_ghost = 'potential ghost probing' in key_actions
    
    # 计算置信度分数
    confidence_score = 0.0
    if high_confidence_ghost:
        confidence_score = 0.9  # 高置信度
    elif potential_ghost:
        confidence_score = 0.6  # 中等置信度
    elif any(keyword in key_actions for keyword in ['emergency', 'braking', 'sudden', 'avoid']):
        confidence_score = 0.3  # 低置信度
    else:
        confidence_score = 0.1  # 基线置信度
    
    # 检测逻辑: 任何形式的鬼探头都算作检测到
    detected = high_confidence_ghost or potential_ghost
    
    return detected, high_confidence_ghost, confidence_score

def evaluate_gpt41_performance(gt_labels: Dict[str, bool], detection_results: List[Dict]) -> Dict:
    """评估GPT-4.1检测性能 (与GPT-4.1评估方法一致)"""
    
    # 准备评估数据
    matched_results = []
    unmatched_videos = []
    
    for result in detection_results:
        video_id = result.get('video_id', '').replace('.avi', '')
        
        # 跳过处理失败的视频
        if 'error' in result:
            continue
            
        if video_id in gt_labels:
            # 提取检测信息
            detected, high_confidence, confidence = extract_ghost_probing_detection(result)
            
            # Ground truth
            ground_truth = gt_labels[video_id]
            
            matched_results.append({
                'video_id': video_id,
                'ground_truth': ground_truth,
                'detected': detected,
                'high_confidence': high_confidence,
                'confidence': confidence,
                'key_actions': result.get('key_actions', ''),
                'summary': result.get('summary', ''),
                'scene_theme': result.get('scene_theme', ''),
                'processing_time': result.get('processing_time', 0)
            })
        else:
            unmatched_videos.append(video_id)
    
    print(f"📊 匹配到 {len(matched_results)} 个有ground truth的检测结果")
    if unmatched_videos:
        print(f"⚠️  {len(unmatched_videos)} 个视频未找到ground truth")
    
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
    
    # 误报率 (False Positive Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 分析检测案例
    true_positives = [r for r in matched_results if r['ground_truth'] and r['detected']]
    false_positives = [r for r in matched_results if not r['ground_truth'] and r['detected']]
    false_negatives = [r for r in matched_results if r['ground_truth'] and not r['detected']]
    true_negatives = [r for r in matched_results if not r['ground_truth'] and not r['detected']]
    
    # 高置信度检测分析
    high_confidence_tp = [r for r in true_positives if r['high_confidence']]
    high_confidence_fp = [r for r in false_positives if r['high_confidence']]
    
    # 计算平均处理时间
    avg_processing_time = np.mean([r['processing_time'] for r in matched_results if r['processing_time'] > 0])
    
    results = {
        'evaluation_summary': {
            'total_videos': total,
            'ground_truth_positives': tp + fn,
            'ground_truth_negatives': tn + fp,
            'detected_positives': tp + fp,
            'detected_negatives': tn + fn
        },
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
            'f1_score': round(f1_score, 4),
            'false_positive_rate': round(fpr, 4)
        },
        'detailed_analysis': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'high_confidence_true_positives': high_confidence_tp,
            'high_confidence_false_positives': high_confidence_fp
        },
        'processing_stats': {
            'avg_processing_time': round(avg_processing_time, 2),
            'total_processing_time': round(sum(r['processing_time'] for r in matched_results), 2)
        },
        'detection_breakdown': {
            'total_ghost_detected': len([r for r in matched_results if r['detected']]),
            'high_confidence_ghost': len([r for r in matched_results if r['high_confidence']]),
            'potential_ghost': len([r for r in matched_results if r['detected'] and not r['high_confidence']])
        }
    }
    
    return results

def print_gpt41_evaluation_report(eval_results: Dict):
    """打印GPT-4.1风格的评估报告"""
    
    print("\n" + "="*100)
    print("🎯 GPT-4.1平衡版100视频鬼探头检测性能评估报告")
    print("="*100)
    
    if 'error' in eval_results:
        print(f"❌ 评估错误: {eval_results['error']}")
        return
    
    # 基本统计信息
    summary = eval_results['evaluation_summary']
    cm = eval_results['confusion_matrix']
    metrics = eval_results['performance_metrics']
    processing = eval_results['processing_stats']
    breakdown = eval_results['detection_breakdown']
    
    print("📊 数据集概览:")
    print(f"   总视频数: {summary['total_videos']}")
    print(f"   Ground Truth鬼探头: {summary['ground_truth_positives']}")
    print(f"   Ground Truth正常: {summary['ground_truth_negatives']}")
    print(f"   检测出鬼探头: {breakdown['total_ghost_detected']}")
    print(f"   高置信度检测: {breakdown['high_confidence_ghost']}")
    print(f"   潜在鬼探头检测: {breakdown['potential_ghost']}")
    print()
    
    # 混淆矩阵 (GPT-4.1格式)
    print("📋 混淆矩阵:")
    print("                预测结果")
    print("              鬼探头  正常")
    print("真实  鬼探头    {:3d}    {:3d}".format(cm['true_positive'], cm['false_negative']))
    print("标签  正常      {:3d}    {:3d}".format(cm['false_positive'], cm['true_negative']))
    print()
    
    # 核心性能指标 (与GPT-4.1对比格式)
    print("🎯 核心性能指标:")
    print(f"   ✅ F1分数:     {metrics['f1_score']:.3f}")
    print(f"   🎯 召回率:     {metrics['recall']:.1%}  ({cm['true_positive']}/{summary['ground_truth_positives']})")
    print(f"   🎯 精确度:     {metrics['precision']:.1%}  ({cm['true_positive']}/{breakdown['total_ghost_detected']})")
    print(f"   📊 准确率:     {metrics['accuracy']:.1%}")
    print(f"   📊 特异性:     {metrics['specificity']:.1%}")
    print(f"   ⚠️  误报率:     {metrics['false_positive_rate']:.1%}")
    print()
    
    # 处理性能
    print("⏱️  处理性能:")
    print(f"   平均处理时间: {processing['avg_processing_time']:.2f}秒/视频")
    print(f"   总处理时间: {processing['total_processing_time']:.2f}秒")
    print()
    
    # 详细案例分析
    detailed = eval_results['detailed_analysis']
    
    print("📈 检测案例分析:")
    
    # 真阳性案例
    if detailed['true_positives']:
        print(f"\n✅ 正确检测的鬼探头 ({len(detailed['true_positives'])}个):")
        for i, tp in enumerate(detailed['true_positives'][:10]):  # 显示前10个
            confidence_str = "高置信度" if tp['high_confidence'] else "潜在"
            print(f"   {i+1:2d}. {tp['video_id']}: {confidence_str} (置信度: {tp['confidence']:.2f})")
            print(f"       关键动作: {tp['key_actions'][:80]}...")
        if len(detailed['true_positives']) > 10:
            print(f"       ... 还有 {len(detailed['true_positives']) - 10} 个")
    
    # 假阳性案例  
    if detailed['false_positives']:
        print(f"\n❌ 误报的正常视频 ({len(detailed['false_positives'])}个):")
        for i, fp in enumerate(detailed['false_positives'][:10]):
            confidence_str = "高置信度" if fp['high_confidence'] else "潜在"
            print(f"   {i+1:2d}. {fp['video_id']}: {confidence_str} (置信度: {fp['confidence']:.2f})")
            print(f"       关键动作: {fp['key_actions'][:80]}...")
        if len(detailed['false_positives']) > 10:
            print(f"       ... 还有 {len(detailed['false_positives']) - 10} 个")
    
    # 假阴性案例
    if detailed['false_negatives']:
        print(f"\n⚠️  漏检的鬼探头 ({len(detailed['false_negatives'])}个):")
        for i, fn in enumerate(detailed['false_negatives'][:10]):
            print(f"   {i+1:2d}. {fn['video_id']}: 置信度: {fn['confidence']:.2f}")
            print(f"       关键动作: {fn['key_actions'][:80]}...")
        if len(detailed['false_negatives']) > 10:
            print(f"       ... 还有 {len(detailed['false_negatives']) - 10} 个")
    
    print("\n" + "="*100)
    
    # 与GPT-4.1基线对比
    print("📊 与GPT-4.1基线对比:")
    print("   GPT-4.1基线性能 (99视频):")
    print("   - F1分数: 0.712")
    print("   - 召回率: 96.3%") 
    print("   - 精确度: 56.5%")
    print("   - 准确率: 57.6%")
    print("   - 误报率: 88.9%")
    print()
    print("   当前测试性能 (100视频):")
    print(f"   - F1分数: {metrics['f1_score']:.3f}")
    print(f"   - 召回率: {metrics['recall']:.1%}")
    print(f"   - 精确度: {metrics['precision']:.1%}")
    print(f"   - 准确率: {metrics['accuracy']:.1%}")
    print(f"   - 误报率: {metrics['false_positive_rate']:.1%}")
    
    print("="*100)

def save_gpt41_format_results(eval_results: Dict, output_path: str):
    """保存GPT-4.1格式的结果"""
    
    # 创建与GPT-4.1一致的结果格式
    gpt41_format = {
        'evaluation_metadata': {
            'model': 'GPT-4.1-Balanced',
            'dataset': 'DADA-100-videos',
            'evaluation_date': datetime.now().isoformat(),
            'ground_truth_source': 'groundtruth_labels.csv',
            'evaluation_method': 'gpt41_compatible'
        },
        'performance_summary': eval_results['performance_metrics'],
        'confusion_matrix': eval_results['confusion_matrix'],
        'dataset_stats': eval_results['evaluation_summary'],
        'detection_breakdown': eval_results['detection_breakdown'],
        'processing_performance': eval_results['processing_stats'],
        'detailed_results': {
            'true_positives': eval_results['detailed_analysis']['true_positives'],
            'false_positives': eval_results['detailed_analysis']['false_positives'],
            'false_negatives': eval_results['detailed_analysis']['false_negatives'],
            'high_confidence_analysis': {
                'true_positives': eval_results['detailed_analysis']['high_confidence_true_positives'],
                'false_positives': eval_results['detailed_analysis']['high_confidence_false_positives']
            }
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gpt41_format, f, indent=2, ensure_ascii=False)
    
    print(f"📄 GPT-4.1格式结果已保存: {output_path}")

def main():
    """主函数"""
    print("🚀 开始评估GPT-4.1平衡版100视频检测结果...")
    
    # 文件路径
    gt_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv"
    
    # 自动查找最新的结果文件
    results_dir = Path("./outputs/results")
    if not results_dir.exists():
        results_dir = Path(".")
    
    gpt41_result_files = list(results_dir.glob("gpt41_balanced_100_videos_*.json"))
    
    if not gpt41_result_files:
        print("❌ 未找到GPT-4.1结果文件")
        print("请确保已运行gpt41_balanced_100_videos.py并生成了结果文件")
        return
    
    # 使用最新的结果文件
    gpt41_results_file = max(gpt41_result_files, key=lambda p: p.stat().st_mtime)
    print(f"📊 使用结果文件: {gpt41_results_file}")
    
    # 检查ground truth文件
    if not Path(gt_file).exists():
        print(f"❌ Ground truth文件不存在: {gt_file}")
        return
    
    # 加载数据
    gt_labels = load_ground_truth_labels(gt_file)
    if not gt_labels:
        print("❌ 无法加载ground truth标签")
        return
    
    gpt41_results = load_gpt41_results(str(gpt41_results_file))
    if not gpt41_results:
        print("❌ 无法加载GPT-4.1检测结果")
        return
    
    # 评估性能
    print("\n🔍 开始性能评估...")
    eval_results = evaluate_gpt41_performance(gt_labels, gpt41_results)
    
    # 打印评估报告
    print_gpt41_evaluation_report(eval_results)
    
    # 保存详细评估结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存原始评估结果
    detailed_output = f"gpt41_evaluation_detailed_{timestamp}.json"
    with open(detailed_output, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    # 保存GPT-4.1兼容格式结果
    gpt41_format_output = f"gpt41_evaluation_summary_{timestamp}.json"
    save_gpt41_format_results(eval_results, gpt41_format_output)
    
    print(f"\n📄 详细评估结果已保存:")
    print(f"   详细结果: {detailed_output}")
    print(f"   GPT-4.1格式: {gpt41_format_output}")

if __name__ == "__main__":
    main()