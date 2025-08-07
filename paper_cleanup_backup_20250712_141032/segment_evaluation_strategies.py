#!/usr/bin/env python3
"""
多段落视频的不同评估策略对比
"""

import os
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_ground_truth():
    """加载Ground Truth标签"""
    labels_file = "result/groundtruth_labels.csv"
    df = pd.read_csv(labels_file, sep='\t')
    
    ground_truth = {}
    for _, row in df.iterrows():
        video_id = row['video_id'].replace('.avi', '')
        label = row['ground_truth_label']
        
        # 解析标签，提取时间信息
        if label == 'none':
            ground_truth[video_id] = {'has_ghost_probing': False, 'time': None}
        else:
            # 例如: "5s: ghost probing" -> 提取时间
            time_part = label.split(':')[0].strip()
            if 's' in time_part:
                time_seconds = int(time_part.replace('s', ''))
                ground_truth[video_id] = {'has_ghost_probing': True, 'time': time_seconds}
            else:
                ground_truth[video_id] = {'has_ghost_probing': True, 'time': None}
    
    return ground_truth

def extract_segment_predictions(result_file):
    """提取每个段落的预测结果"""
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        segment_predictions = []
        for segment in segments:
            if isinstance(segment, dict):
                key_actions = segment.get('key_actions', '').lower()
                start_time = segment.get('Start_Timestamp', '0.0s')
                end_time = segment.get('End_Timestamp', '10.0s')
                
                # 提取时间
                start_sec = float(start_time.replace('s', ''))
                end_sec = float(end_time.replace('s', ''))
                
                has_ghost_probing = 'ghost probing' in key_actions
                
                segment_predictions.append({
                    'start_time': start_sec,
                    'end_time': end_sec,
                    'has_ghost_probing': has_ghost_probing,
                    'key_actions': segment.get('key_actions', '')
                })
        
        return segment_predictions
    except Exception as e:
        print(f"❌ 解析文件失败: {result_file}")
        return []

def strategy_1_any_segment(segment_predictions):
    """策略1：任何段落有鬼探头就算整个视频有鬼探头"""
    return any(seg['has_ghost_probing'] for seg in segment_predictions)

def strategy_2_all_segments(segment_predictions):
    """策略2：所有段落都有鬼探头才算整个视频有鬼探头"""
    if not segment_predictions:
        return False
    return all(seg['has_ghost_probing'] for seg in segment_predictions)

def strategy_3_majority_vote(segment_predictions):
    """策略3：多数投票决定"""
    if not segment_predictions:
        return False
    ghost_count = sum(seg['has_ghost_probing'] for seg in segment_predictions)
    return ghost_count > len(segment_predictions) / 2

def strategy_4_time_based(segment_predictions, ground_truth_time):
    """策略4：基于时间匹配的策略"""
    if ground_truth_time is None:
        # 如果没有具体时间，使用任何段落策略
        return strategy_1_any_segment(segment_predictions)
    
    # 找到包含ground truth时间的段落
    for seg in segment_predictions:
        if seg['start_time'] <= ground_truth_time <= seg['end_time']:
            return seg['has_ghost_probing']
    
    # 如果没有找到对应时间段，使用任何段落策略
    return strategy_1_any_segment(segment_predictions)

def evaluate_strategies(model_dir, ground_truth):
    """评估不同策略的性能"""
    
    if not os.path.exists(model_dir):
        return None
    
    results = {
        'strategy_1_any': {'predictions': [], 'true_labels': []},
        'strategy_2_all': {'predictions': [], 'true_labels': []}, 
        'strategy_3_majority': {'predictions': [], 'true_labels': []},
        'strategy_4_time': {'predictions': [], 'true_labels': []}
    }
    
    processed_count = 0
    error_count = 0
    
    for video_id in ground_truth.keys():
        result_file = os.path.join(model_dir, f"actionSummary_{video_id}.json")
        if not os.path.exists(result_file):
            continue
        
        segment_predictions = extract_segment_predictions(result_file)
        if not segment_predictions:
            error_count += 1
            continue
        
        processed_count += 1
        gt = ground_truth[video_id]
        true_label = gt['has_ghost_probing']
        gt_time = gt['time']
        
        # 应用不同策略
        pred_1 = strategy_1_any_segment(segment_predictions)
        pred_2 = strategy_2_all_segments(segment_predictions)
        pred_3 = strategy_3_majority_vote(segment_predictions)
        pred_4 = strategy_4_time_based(segment_predictions, gt_time)
        
        # 记录结果
        for strategy, pred in [('strategy_1_any', pred_1), ('strategy_2_all', pred_2), 
                              ('strategy_3_majority', pred_3), ('strategy_4_time', pred_4)]:
            results[strategy]['predictions'].append(pred)
            results[strategy]['true_labels'].append(true_label)
    
    print(f"📊 处理了 {processed_count} 个视频，{error_count} 个错误")
    
    # 计算指标
    strategy_metrics = {}
    for strategy_name, data in results.items():
        if len(data['predictions']) > 0:
            preds = [1 if p else 0 for p in data['predictions']]
            labels = [1 if l else 0 for l in data['true_labels']]
            
            strategy_metrics[strategy_name] = {
                'accuracy': accuracy_score(labels, preds),
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'f1': f1_score(labels, preds, zero_division=0),
                'video_count': len(preds)
            }
    
    return strategy_metrics

def print_strategy_comparison(model_name, strategy_metrics):
    """打印策略对比结果"""
    print(f"\n📊 {model_name} - 不同段落评估策略对比")
    print("=" * 100)
    
    print(f"{'策略':<20} {'准确率':<10} {'精确度':<10} {'召回率':<10} {'F1分数':<10} {'视频数':<10}")
    print("-" * 100)
    
    strategy_names = {
        'strategy_1_any': '任何段落',
        'strategy_2_all': '所有段落',
        'strategy_3_majority': '多数投票',
        'strategy_4_time': '时间匹配'
    }
    
    for strategy_key, strategy_name in strategy_names.items():
        if strategy_key in strategy_metrics:
            metrics = strategy_metrics[strategy_key]
            print(f"{strategy_name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
                  f"{metrics['recall']:<10.3f} {metrics['f1']:<10.3f} {metrics['video_count']:<10}")
    
    # 找出最佳策略
    best_f1_strategy = max(strategy_metrics.keys(), key=lambda k: strategy_metrics[k]['f1'])
    best_f1_name = strategy_names[best_f1_strategy]
    best_f1_score = strategy_metrics[best_f1_strategy]['f1']
    
    print(f"\n🏆 最佳策略: {best_f1_name} (F1 = {best_f1_score:.3f})")

def analyze_specific_cases(model_dir, ground_truth):
    """分析具体案例"""
    print(f"\n🔍 具体案例分析:")
    print("-" * 80)
    
    cases_analyzed = 0
    for video_id in list(ground_truth.keys())[:5]:  # 分析前5个视频
        result_file = os.path.join(model_dir, f"actionSummary_{video_id}.json")
        if not os.path.exists(result_file):
            continue
        
        segment_predictions = extract_segment_predictions(result_file)
        if not segment_predictions:
            continue
        
        gt = ground_truth[video_id]
        print(f"\n📹 {video_id}:")
        print(f"   Ground Truth: {'鬼探头' if gt['has_ghost_probing'] else '正常'}", end="")
        if gt['time']:
            print(f" (时间: {gt['time']}s)")
        else:
            print()
        
        print(f"   段落数: {len(segment_predictions)}")
        for i, seg in enumerate(segment_predictions):
            status = "🔴 鬼探头" if seg['has_ghost_probing'] else "🟢 正常"
            print(f"   段落{i+1} ({seg['start_time']:.1f}s-{seg['end_time']:.1f}s): {status}")
        
        # 不同策略的结果
        pred_1 = strategy_1_any_segment(segment_predictions)
        pred_2 = strategy_2_all_segments(segment_predictions)
        pred_3 = strategy_3_majority_vote(segment_predictions)
        pred_4 = strategy_4_time_based(segment_predictions, gt['time'])
        
        print(f"   策略结果: 任何段落={pred_1}, 所有段落={pred_2}, 多数投票={pred_3}, 时间匹配={pred_4}")
        
        cases_analyzed += 1
        if cases_analyzed >= 5:
            break

def main():
    print("🔧 多段落视频评估策略分析")
    print("=" * 80)
    
    # 加载Ground Truth
    ground_truth = load_ground_truth()
    print(f"📋 Ground Truth总数: {len(ground_truth)}")
    
    # 统计有时间信息的标签
    with_time = sum(1 for gt in ground_truth.values() if gt['time'] is not None)
    print(f"📋 有具体时间信息的标签: {with_time}/{len(ground_truth)}")
    
    # 分析平衡版GPT-4.1
    model_dir = "result/gpt41-balanced-full"
    if os.path.exists(model_dir):
        print(f"\n🔍 分析平衡版GPT-4.1的段落评估策略...")
        
        strategy_metrics = evaluate_strategies(model_dir, ground_truth)
        if strategy_metrics:
            print_strategy_comparison("平衡版GPT-4.1", strategy_metrics)
            analyze_specific_cases(model_dir, ground_truth)
        else:
            print("❌ 无法获取策略评估结果")
    else:
        print("❌ 平衡版结果目录不存在")

if __name__ == "__main__":
    main()