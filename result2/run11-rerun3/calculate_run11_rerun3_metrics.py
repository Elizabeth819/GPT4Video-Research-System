#!/usr/bin/env python3
"""
Calculate performance metrics for Run 11 Rerun3
"""
import json
from pathlib import Path

def calculate_metrics():
    results_file = Path("/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run11-rerun3/run11_gpt41_rerun3_final_results_20250730_102243.json")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count results
    tp = fp = tn = fn = 0
    
    for result in data['detailed_results']:
        evaluation = result['evaluation']
        if evaluation == 'TP':
            tp += 1
        elif evaluation == 'FP':
            fp += 1
        elif evaluation == 'TN':
            tn += 1
        elif evaluation == 'FN':
            fn += 1
    
    total = tp + fp + tn + fn
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    print("🚀 Run 11 Rerun3 Performance Metrics")
    print("="*50)
    print(f"📊 Total Videos: {total}")
    print(f"🎯 True Positives (TP): {tp}")
    print(f"🔴 False Positives (FP): {fp}")
    print(f"✅ True Negatives (TN): {tn}")
    print(f"🟡 False Negatives (FN): {fn}")
    print()
    print("📈 Performance Metrics:")
    print(f"🎯 F1-Score: {f1:.3f}")
    print(f"🔍 Precision: {precision:.3f}")
    print(f"📡 Recall: {recall:.3f}")
    print(f"🛡️ Specificity: {specificity:.3f}")
    print(f"✅ Accuracy: {accuracy:.3f}")
    print(f"⚖️ Balanced Accuracy: {balanced_accuracy:.3f}")
    
    # Model info
    exp_info = data['experiment_info']
    print(f"\n🔧 Model: {exp_info['model']}")
    print(f"📝 Prompt: {exp_info['prompt_version']}")
    print(f"📅 Timestamp: {exp_info['timestamp']}")
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'f1_score': f1, 'precision': precision, 'recall': recall,
        'specificity': specificity, 'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'total_videos': total
    }

if __name__ == "__main__":
    metrics = calculate_metrics()