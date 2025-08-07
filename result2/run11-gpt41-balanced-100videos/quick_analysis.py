#!/usr/bin/env python3
"""
Run 11 快速分析脚本 - 分析当前进度和性能
"""

import json
import pandas as pd

def analyze_current_progress():
    """分析当前进度"""
    intermediate_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run11-gpt41-balanced-100videos/run11_intermediate_updated_20250727_155049.json"
    
    with open(intermediate_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    processed_videos = len(results['detailed_results'])
    
    # 计算当前性能
    tp = fp = tn = fn = errors = 0
    
    for result in results['detailed_results']:
        if result['status'] == 'error' or result['status'] == 'parse_error':
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
    
    print(f"""
=== Run 11: GPT-4.1+Balanced 当前进度分析 ===

处理进度: {processed_videos}/100 ({processed_videos}%)

当前性能指标:
- F1分数: {f1:.3f}
- 精确度: {precision:.3f} 
- 召回率: {recall:.3f}
- 准确率: {accuracy:.3f}

混淆矩阵:
- TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}
- 错误: {errors}

与历史目标对比:
- F1分数: {f1:.3f} vs 0.712 ({f1-0.712:+.3f})
- 召回率: {recall:.3f} vs 0.963 ({recall-0.963:+.3f})
- 精确度: {precision:.3f} vs 0.565 ({precision-0.565:+.3f})

复现趋势: {'✅ 有希望复现' if f1 >= 0.6 else '❓ 需要观察'}
""")

if __name__ == "__main__":
    analyze_current_progress()