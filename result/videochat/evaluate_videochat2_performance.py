#!/usr/bin/env python3
"""
VideoChat2 Ghost Probing Detection Evaluation Script
Compares VideoChat2 predictions against ground truth labels
Calculates accuracy, precision, recall, and F1-score metrics
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re

def load_ground_truth_labels(csv_path: str) -> Dict[str, str]:
    """Load ground truth labels from CSV file"""
    ground_truth = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            video_id = row['video_id'].replace('.avi', '')
            label = row['ground_truth_label']
            
            # Convert ground truth to binary classification
            if label == 'none':
                ground_truth[video_id] = 'normal_traffic'
            elif 'ghost probing' in label:
                ground_truth[video_id] = 'ghost_probing'
            elif label == 'cut-in' or 'cut-in' in label:
                # Treat cut-in as normal traffic for this binary classification
                ground_truth[video_id] = 'normal_traffic'
            else:
                # Skip videos with unclear labels
                continue
    
    return ground_truth

def extract_videochat2_predictions(results_dir: str) -> Dict[str, str]:
    """Extract VideoChat2 predictions from individual JSON files"""
    predictions = {}
    
    results_path = Path(results_dir)
    
    for json_file in results_path.glob("actionSummary_images_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                segment = data[0]
                
                # Extract video ID from filename
                filename = json_file.stem
                match = re.search(r'actionSummary_images_(\d+)_(\d+)', filename)
                if match:
                    category, number = match.groups()
                    video_id = f"images_{category}_{number}"
                    
                    # Determine prediction based on sentiment and key_actions
                    sentiment = segment.get('sentiment', '')
                    key_actions = segment.get('key_actions', '')
                    scene_theme = segment.get('scene_theme', '')
                    
                    # VideoChat2 classification logic based on generated format
                    if (sentiment == 'Negative' and 
                        scene_theme == 'Dramatic' and 
                        'ghost probing' in key_actions.lower()):
                        predictions[video_id] = 'ghost_probing'
                    else:
                        predictions[video_id] = 'normal_traffic'
                        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return predictions

def calculate_metrics(ground_truth: Dict[str, str], predictions: Dict[str, str]) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, and F1-score"""
    
    # Find common video IDs
    common_ids = set(ground_truth.keys()) & set(predictions.keys())
    
    if not common_ids:
        raise ValueError("No common video IDs found between ground truth and predictions")
    
    # Calculate confusion matrix
    tp = fp = tn = fn = 0
    
    correct_predictions = []
    incorrect_predictions = []
    
    for video_id in common_ids:
        gt_label = ground_truth[video_id]
        pred_label = predictions[video_id]
        
        if gt_label == 'ghost_probing' and pred_label == 'ghost_probing':
            tp += 1
            correct_predictions.append((video_id, gt_label, pred_label))
        elif gt_label == 'normal_traffic' and pred_label == 'ghost_probing':
            fp += 1
            incorrect_predictions.append((video_id, gt_label, pred_label))
        elif gt_label == 'normal_traffic' and pred_label == 'normal_traffic':
            tn += 1
            correct_predictions.append((video_id, gt_label, pred_label))
        elif gt_label == 'ghost_probing' and pred_label == 'normal_traffic':
            fn += 1
            incorrect_predictions.append((video_id, gt_label, pred_label))
    
    # Calculate metrics
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'total_samples': total,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': incorrect_predictions
    }

def generate_detailed_report(metrics: Dict, ground_truth: Dict[str, str], 
                           predictions: Dict[str, str]) -> str:
    """Generate detailed evaluation report"""
    
    report = []
    report.append("=" * 80)
    report.append("VideoChat2 Ghost Probing Detection Evaluation Report")
    report.append("=" * 80)
    report.append("")
    
    # Overall metrics
    report.append("OVERALL PERFORMANCE METRICS:")
    report.append("-" * 40)
    report.append(f"Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    report.append(f"Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    report.append(f"Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    report.append(f"F1-Score:  {metrics['f1_score']:.3f}")
    report.append("")
    
    # Confusion matrix
    report.append("CONFUSION MATRIX:")
    report.append("-" * 40)
    report.append("                    Predicted")
    report.append("                Ghost    Normal")
    report.append(f"Actual Ghost    {metrics['true_positives']:5d}    {metrics['false_negatives']:5d}")
    report.append(f"       Normal   {metrics['false_positives']:5d}    {metrics['true_negatives']:5d}")
    report.append("")
    
    # Ground truth distribution
    gt_ghost = sum(1 for label in ground_truth.values() if label == 'ghost_probing')
    gt_normal = sum(1 for label in ground_truth.values() if label == 'normal_traffic')
    
    report.append("GROUND TRUTH DISTRIBUTION:")
    report.append("-" * 40)
    report.append(f"Ghost Probing videos: {gt_ghost}")
    report.append(f"Normal Traffic videos: {gt_normal}")
    report.append(f"Total videos: {gt_ghost + gt_normal}")
    report.append("")
    
    # VideoChat2 predictions distribution
    pred_ghost = sum(1 for label in predictions.values() if label == 'ghost_probing')
    pred_normal = sum(1 for label in predictions.values() if label == 'normal_traffic')
    
    report.append("VIDEOCHAT2 PREDICTIONS:")
    report.append("-" * 40)
    report.append(f"Predicted Ghost Probing: {pred_ghost}")
    report.append(f"Predicted Normal Traffic: {pred_normal}")
    report.append("")
    
    # Videos classified as NOT ghost probing (normal traffic)
    normal_predictions = [vid for vid, pred in predictions.items() if pred == 'normal_traffic']
    report.append("VIDEOS CLASSIFIED AS NOT GHOST PROBING (Normal Traffic):")
    report.append("-" * 60)
    
    for video_id in sorted(normal_predictions):
        gt_label = ground_truth.get(video_id, 'unknown')
        status = "✓ Correct" if gt_label == 'normal_traffic' else "✗ Incorrect"
        report.append(f"{video_id:20s} | GT: {gt_label:15s} | {status}")
    
    report.append("")
    
    # Detailed analysis
    report.append("DETAILED ANALYSIS:")
    report.append("-" * 40)
    
    # False positives (predicted ghost but actually normal)
    false_positives = [(vid, gt, pred) for vid, gt, pred in metrics['incorrect_predictions'] 
                      if pred == 'ghost_probing' and gt == 'normal_traffic']
    
    if false_positives:
        report.append(f"\nFALSE POSITIVES ({len(false_positives)} videos):")
        report.append("Videos incorrectly classified as ghost probing:")
        for vid, gt, pred in false_positives:
            report.append(f"  {vid}: GT={gt}, Predicted={pred}")
    
    # False negatives (predicted normal but actually ghost)
    false_negatives = [(vid, gt, pred) for vid, gt, pred in metrics['incorrect_predictions'] 
                      if pred == 'normal_traffic' and gt == 'ghost_probing']
    
    if false_negatives:
        report.append(f"\nFALSE NEGATIVES ({len(false_negatives)} videos):")
        report.append("Ghost probing videos missed by VideoChat2:")
        for vid, gt, pred in false_negatives:
            report.append(f"  {vid}: GT={gt}, Predicted={pred}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def get_100_video_dataset() -> List[str]:
    """Get the 100-video dataset used for VideoChat2 evaluation"""
    videos = []
    
    # DADA-100-videos dataset mapping
    # images_1_001 to images_1_027 (27 videos)
    for i in range(1, 28):
        videos.append(f"images_1_{i:03d}")
    
    # images_2_001 to images_2_005 (5 videos) 
    for i in range(1, 6):
        videos.append(f"images_2_{i:03d}")
    
    # images_3_001 to images_3_007 (7 videos)
    for i in range(1, 8):
        videos.append(f"images_3_{i:03d}")
    
    # images_4_001 to images_4_008 (8 videos)
    for i in range(1, 9):
        videos.append(f"images_4_{i:03d}")
    
    # images_5_001 to images_5_053 (53 videos)
    for i in range(1, 54):
        videos.append(f"images_5_{i:03d}")
    
    return videos

def main():
    """Main evaluation function"""
    
    # File paths
    ground_truth_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv"
    videochat2_results_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/videochat/blue_jewel_results/artifacts/outputs"
    
    print("Loading ground truth labels...")
    all_ground_truth = load_ground_truth_labels(ground_truth_path)
    
    # Filter to only the 100 videos in our dataset
    video_list = get_100_video_dataset()
    ground_truth = {vid: all_ground_truth[vid] for vid in video_list if vid in all_ground_truth}
    print(f"Loaded {len(ground_truth)} ground truth labels for 100-video dataset")
    
    print("Extracting VideoChat2 predictions...")
    all_predictions = extract_videochat2_predictions(videochat2_results_dir)
    
    # Filter to only the 100 videos in our dataset
    predictions = {vid: all_predictions[vid] for vid in video_list if vid in all_predictions}
    print(f"Extracted {len(predictions)} VideoChat2 predictions for 100-video dataset")
    
    print("Calculating performance metrics...")
    metrics = calculate_metrics(ground_truth, predictions)
    
    print("Generating detailed report...")
    report = generate_detailed_report(metrics, ground_truth, predictions)
    
    # Save report to file
    report_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/videochat/videochat2_evaluation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Evaluation report saved to: {report_file}")
    print("\n" + "="*60)
    print("QUICK SUMMARY:")
    print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"Recall: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print("="*60)
    
    # Show videos classified as NOT ghost probing
    normal_predictions = [vid for vid, pred in predictions.items() if pred == 'normal_traffic']
    print(f"\nVideos classified as NOT GHOST PROBING: {len(normal_predictions)} videos")
    for video_id in sorted(normal_predictions)[:10]:  # Show first 10
        gt_label = ground_truth.get(video_id, 'unknown')
        print(f"  {video_id}: Ground Truth = {gt_label}")
    
    if len(normal_predictions) > 10:
        print(f"  ... and {len(normal_predictions) - 10} more (see full report)")

if __name__ == "__main__":
    main()