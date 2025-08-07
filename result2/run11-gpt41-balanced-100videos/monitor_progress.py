#!/usr/bin/env python3
"""
Run 11 进度监控脚本
"""

import json
import os
import glob
import time
from datetime import datetime

def get_latest_intermediate_file():
    """获取最新的中间结果文件"""
    pattern = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run11-gpt41-balanced-100videos/run11_intermediate*.json"
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def analyze_progress(file_path):
    """分析进度"""
    if not file_path or not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 获取已处理视频
    processed_videos = []
    for result in results['detailed_results']:
        processed_videos.append(result['video_id'])
    
    unique_videos = list(set(processed_videos))
    unique_videos.sort()
    
    # 计算性能指标
    tp = fp = tn = fn = errors = 0
    for result in results['detailed_results']:
        if result['status'] in ['error', 'parse_error']:
            errors += 1
            continue
            
        predicted = result['predicted_label']
        actual = result['actual_label']
        
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 1:
            fn += 1
        else:
            tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'processed_count': len(unique_videos),
        'latest_video': unique_videos[-1] if unique_videos else 'N/A',
        'progress_percent': len(unique_videos),
        'metrics': {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'errors': errors
        },
        'file_modified_time': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
    }

def main():
    """主监控循环"""
    print("🔍 Run 11: GPT-4.1+Balanced 进度监控")
    print("=" * 60)
    
    latest_file = get_latest_intermediate_file()
    if not latest_file:
        print("❌ 未找到中间结果文件")
        return
    
    progress = analyze_progress(latest_file)
    if not progress:
        print("❌ 无法分析进度")
        return
    
    print(f"📁 文件: {os.path.basename(latest_file)}")
    print(f"🕒 更新时间: {progress['file_modified_time']}")
    print(f"📊 进度: {progress['processed_count']}/100 ({progress['processed_count']}%)")
    print(f"🎯 最新视频: {progress['latest_video']}")
    print()
    
    metrics = progress['metrics']
    print("📈 当前性能指标:")
    print(f"  F1分数: {metrics['f1']:.3f}")
    print(f"  精确度: {metrics['precision']:.3f}")
    print(f"  召回率: {metrics['recall']:.3f}")
    print(f"  准确率: {metrics['accuracy']:.3f}")
    print()
    
    print("🔢 混淆矩阵:")
    print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}")
    print(f"  TN: {metrics['tn']}, FN: {metrics['fn']}")
    print(f"  错误: {metrics['errors']}")
    print()
    
    print("🎯 与历史目标对比:")
    print(f"  F1分数: {metrics['f1']:.3f} vs 0.712 ({metrics['f1']-0.712:+.3f})")
    print(f"  召回率: {metrics['recall']:.3f} vs 0.963 ({metrics['recall']-0.963:+.3f})")
    print(f"  精确度: {metrics['precision']:.3f} vs 0.565 ({metrics['precision']-0.565:+.3f})")
    print()
    
    # 复现趋势评估
    if progress['processed_count'] >= 25:
        if metrics['f1'] >= 0.6:
            trend = "✅ 有希望复现"
        elif metrics['f1'] >= 0.4:
            trend = "⚠️ 部分复现"
        else:
            trend = "❌ 复现困难"
    else:
        trend = "📊 数据不足，继续观察"
    
    print(f"📈 复现趋势: {trend}")
    print("=" * 60)

if __name__ == "__main__":
    main()