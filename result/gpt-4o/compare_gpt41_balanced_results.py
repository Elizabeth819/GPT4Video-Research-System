#!/usr/bin/env python3
"""
Compare GPT-4.1 Balanced results with ground truth and generate metrics
Compatible with existing GPT-4.1 analysis format
"""

import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_groundtruth(file_path):
    """Load ground truth from CSV file"""
    groundtruth = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            groundtruth[row['video_id']] = row['ground_truth_label']
    return groundtruth

def parse_predictions(results_file):
    """Parse predictions from GPT-4.1 results JSON"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    for result in data.get('results', []):
        video_id = result['video_id']
        key_actions = result.get('key_actions', '')
        
        # Determine if ghost probing was detected
        if 'ghost probing' in key_actions.lower():
            predictions[video_id] = 'ghost_probing'
        else:
            predictions[video_id] = 'normal'
    
    return predictions

def create_binary_labels(groundtruth, predictions):
    """Create binary labels for comparison"""
    y_true = []
    y_pred = []
    
    for video_id in groundtruth.keys():
        # Ground truth: 1 if ghost probing, 0 if normal
        gt_label = groundtruth[video_id]
        y_true.append(1 if 'ghost probing' in gt_label.lower() else 0)
        
        # Prediction: 1 if ghost probing detected, 0 if normal
        pred_label = predictions.get(video_id, 'normal')
        y_pred.append(1 if pred_label == 'ghost_probing' else 0)
    
    return y_true, y_pred

def calculate_metrics(y_true, y_pred):
    """Calculate detailed metrics"""
    # Basic metrics
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    
    # Calculate rates
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'false_positive_rate': fpr,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'total_samples': len(y_true)
    }

def generate_confusion_matrix_plot(y_true, y_pred, save_path=None):
    """Generate confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Ghost Probing'],
                yticklabels=['Normal', 'Ghost Probing'])
    plt.title('GPT-4.1 Balanced Ghost Probing Detection\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(groundtruth, predictions, results_file):
    """Generate detailed comparison report"""
    
    # Create binary labels
    y_true, y_pred = create_binary_labels(groundtruth, predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Load full results for additional analysis
    with open(results_file, 'r') as f:
        full_results = json.load(f)
    
    # Generate report
    report = {
        'experiment_info': {
            'model': 'GPT-4.1 Balanced Prompt',
            'dataset': 'DADA-2000 Categories 1-5',
            'total_videos': len(groundtruth),
            'analysis_date': datetime.now().isoformat()
        },
        'performance_metrics': metrics,
        'detailed_analysis': {
            'ghost_probing_videos': sum(y_true),
            'normal_videos': len(y_true) - sum(y_true),
            'detected_ghost_probing': sum(y_pred),
            'detected_normal': len(y_pred) - sum(y_pred)
        },
        'comparison_with_balanced_target': {
            'target_recall': 0.963,
            'actual_recall': metrics['recall'],
            'target_precision': 0.565,
            'actual_precision': metrics['precision'],
            'target_f1': 0.712,
            'actual_f1': metrics['f1_score'],
            'recall_difference': metrics['recall'] - 0.963,
            'precision_difference': metrics['precision'] - 0.565,
            'f1_difference': metrics['f1_score'] - 0.712
        }
    }
    
    return report

def create_comparison_csv(groundtruth, predictions, results_file, output_path):
    """Create detailed comparison CSV"""
    
    # Load full results
    with open(results_file, 'r') as f:
        full_results = json.load(f)
    
    # Create comparison data
    comparison_data = []
    
    for result in full_results.get('results', []):
        video_id = result['video_id']
        gt_label = groundtruth.get(video_id, 'none')
        
        # Parse ground truth
        has_ghost_probing = 'ghost probing' in gt_label.lower()
        ghost_time = None
        if has_ghost_probing:
            match = re.search(r'(\d+)s:', gt_label)
            if match:
                ghost_time = int(match.group(1))
        
        # Parse prediction
        key_actions = result.get('key_actions', '')
        detected_ghost_probing = 'ghost probing' in key_actions.lower()
        
        # Determine correctness
        correct = (detected_ghost_probing and has_ghost_probing) or (not detected_ghost_probing and not has_ghost_probing)
        
        comparison_data.append({
            'video_id': video_id,
            'ground_truth': gt_label,
            'has_ghost_probing': has_ghost_probing,
            'ghost_time': ghost_time,
            'prediction': key_actions,
            'detected_ghost_probing': detected_ghost_probing,
            'confidence': result.get('detection_confidence', 0),
            'detection_time': result.get('detection_time', None),
            'correct': correct,
            'prediction_type': 'TP' if (detected_ghost_probing and has_ghost_probing) else
                             'FP' if (detected_ghost_probing and not has_ghost_probing) else
                             'FN' if (not detected_ghost_probing and has_ghost_probing) else 'TN'
        })
    
    # Save to CSV
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_path, index=False)
    
    return comparison_data

def main():
    """Main analysis function"""
    
    print("🔍 GPT-4.1 Balanced Ghost Probing Analysis")
    print("="*50)
    
    # File paths
    groundtruth_file = 'groundtruth_labels.csv'
    results_file = 'gpt41_balanced_ghost_probing_results.json'
    
    # Check if files exist
    import os
    if not os.path.exists(groundtruth_file):
        print(f"❌ Ground truth file not found: {groundtruth_file}")
        return
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return
    
    # Load data
    print("📊 Loading data...")
    groundtruth = load_groundtruth(groundtruth_file)
    predictions = parse_predictions(results_file)
    
    print(f"✅ Loaded {len(groundtruth)} ground truth labels")
    print(f"✅ Loaded {len(predictions)} predictions")
    
    # Generate analysis
    print("🔍 Generating analysis...")
    report = generate_detailed_report(groundtruth, predictions, results_file)
    
    # Create comparison CSV
    print("📋 Creating comparison CSV...")
    comparison_data = create_comparison_csv(groundtruth, predictions, results_file, 'gpt41_balanced_comparison.csv')
    
    # Generate confusion matrix
    print("📊 Generating confusion matrix...")
    y_true, y_pred = create_binary_labels(groundtruth, predictions)
    generate_confusion_matrix_plot(y_true, y_pred, 'gpt41_balanced_confusion_matrix.png')
    
    # Save report
    with open('gpt41_balanced_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 GPT-4.1 Balanced Performance Summary")
    print("="*50)
    
    metrics = report['performance_metrics']
    print(f"📹 Total Videos: {metrics['total_samples']}")
    print(f"🎯 Accuracy: {metrics['accuracy']:.1%}")
    print(f"🔍 Precision: {metrics['precision']:.1%}")
    print(f"📈 Recall: {metrics['recall']:.1%}")
    print(f"⚖️  F1 Score: {metrics['f1_score']:.3f}")
    print(f"❌ False Positive Rate: {metrics['false_positive_rate']:.1%}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    print(f"\n🎯 Comparison with Balanced Target:")
    comparison = report['comparison_with_balanced_target']
    print(f"Recall: {comparison['actual_recall']:.1%} vs {comparison['target_recall']:.1%} (diff: {comparison['recall_difference']:+.1%})")
    print(f"Precision: {comparison['actual_precision']:.1%} vs {comparison['target_precision']:.1%} (diff: {comparison['precision_difference']:+.1%})")
    print(f"F1 Score: {comparison['actual_f1']:.3f} vs {comparison['target_f1']:.3f} (diff: {comparison['f1_difference']:+.3f})")
    
    print("\n📁 Output Files:")
    print("- gpt41_balanced_analysis_report.json")
    print("- gpt41_balanced_comparison.csv")
    print("- gpt41_balanced_confusion_matrix.png")
    
    print("\n✅ Analysis completed successfully!")

if __name__ == '__main__':
    main()