#!/usr/bin/env python3
"""
Run 8 前30个视频性能分析
分析目前已完成的30个视频的性能指标
"""

import json
from collections import Counter

def analyze_30videos_performance():
    """分析前30个视频的性能"""
    
    # 读取30个视频的中间结果
    with open('/Users/wanmeng/repository/GPT4Video-cobra-auto/VIP/run8_ghost_probing_100videos_results/run8_intermediate_30videos_20250727_092402.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    detailed_results = results['detailed_results']
    
    print(f"🎯 Run 8 前30个视频性能分析")
    print("=" * 60)
    
    # 提取评估结果
    evaluations = [r['evaluation'] for r in detailed_results]
    eval_counts = Counter(evaluations)
    
    tp = eval_counts.get('TP', 0)
    fp = eval_counts.get('FP', 0) 
    tn = eval_counts.get('TN', 0)
    fn = eval_counts.get('FN', 0)
    errors = eval_counts.get('ERROR', 0)
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print(f"📊 性能指标 (前30个视频):")
    print(f"   精确度: {precision:.3f} ({precision*100:.1f}%)")
    print(f"   召回率: {recall:.3f} ({recall*100:.1f}%)")
    print(f"   F1分数: {f1:.3f} ({f1*100:.1f}%)")
    print(f"   准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print(f"\n📈 混淆矩阵:")
    print(f"   TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, ERROR: {errors}")
    
    # 详细结果列表
    print(f"\n📋 详细结果:")
    print(f"{'序号':<4} {'视频ID':<18} {'Ground Truth':<15} {'检测结果':<12} {'评估':<6}")
    print("-" * 65)
    
    for i, result in enumerate(detailed_results, 1):
        video_id = result['video_id']
        gt = result['ground_truth']
        pred = result['key_actions']
        eval_result = result['evaluation']
        
        print(f"{i:<4} {video_id:<18} {gt:<15} {pred:<12} {eval_result:<6}")
    
    # 与之前Run 7 Enhanced (20视频)的对比
    print(f"\n🔍 与Run 7 Enhanced (20视频)对比:")
    print(f"   Run 7 Enhanced F1: 0.774 (77.4%)")
    print(f"   Run 8 前30视频F1: {f1:.3f} ({f1*100:.1f}%)")
    
    if f1 > 0.774:
        print(f"   ✅ Run 8表现更好，提升 {(f1-0.774)*100:.1f}%")
    elif abs(f1 - 0.774) < 0.01:
        print(f"   ≈ 性能基本相当")
    else:
        print(f"   ⚠️ Run 8表现略低，下降 {(0.774-f1)*100:.1f}%")
    
    # 趋势分析
    print(f"\n📈 趋势分析:")
    print(f"   完成视频数: {len(detailed_results)}/100")
    print(f"   预计剩余时间: 约{(100-len(detailed_results)) * 15 // 60}分钟")
    print(f"   当前性能稳定性: {'良好' if f1 > 0.7 else '需要观察'}")

if __name__ == "__main__":
    analyze_30videos_performance()