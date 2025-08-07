#!/usr/bin/env python3
"""
Comprehensive VideoChat2 Analysis Script
Analyzes VideoChat2's actual behavior pattern and provides detailed evaluation
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re
from collections import Counter

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

def analyze_videochat2_behavior(results_dir: str) -> Dict:
    """Analyze VideoChat2's actual behavior patterns"""
    results_path = Path(results_dir)
    
    sentiment_counts = Counter()
    scene_theme_counts = Counter()
    key_actions_patterns = Counter()
    all_results = {}
    
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
                    
                    sentiment = segment.get('sentiment', '')
                    scene_theme = segment.get('scene_theme', '')
                    key_actions = segment.get('key_actions', '')
                    
                    sentiment_counts[sentiment] += 1
                    scene_theme_counts[scene_theme] += 1
                    
                    # Analyze key actions for ghost probing mentions
                    if 'ghost probing' in key_actions.lower():
                        key_actions_patterns['ghost_probing_mentioned'] += 1
                    else:
                        key_actions_patterns['no_ghost_probing'] += 1
                    
                    all_results[video_id] = {
                        'sentiment': sentiment,
                        'scene_theme': scene_theme,
                        'key_actions': key_actions,
                        'summary': segment.get('summary', ''),
                        'actions': segment.get('actions', ''),
                        'file': str(json_file)
                    }
                        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return {
        'sentiment_distribution': dict(sentiment_counts),
        'scene_theme_distribution': dict(scene_theme_counts),
        'key_actions_patterns': dict(key_actions_patterns),
        'total_files': len(all_results),
        'all_results': all_results
    }

def corrected_videochat2_classification(results_dir: str) -> Dict[str, str]:
    """Apply corrected classification based on VideoChat2's actual behavior"""
    analysis = analyze_videochat2_behavior(results_dir)
    predictions = {}
    
    # Based on the analysis, VideoChat2 seems to classify everything as ghost probing
    # Let's use a more nuanced approach based on the content analysis
    
    for video_id, result in analysis['all_results'].items():
        # Apply corrected logic: only classify as ghost probing if there's actual evidence
        key_actions = result['key_actions'].lower()
        summary = result['summary'].lower()
        actions = result['actions'].lower()
        
        # Look for genuine ghost probing indicators
        ghost_indicators = [
            'sudden appearance',
            'hidden from view',
            'unpredictable trajectory',
            'emergency braking',
            'collision risk',
            'appears suddenly',
            'concealed position'
        ]
        
        # Count evidence of ghost probing
        evidence_count = sum(1 for indicator in ghost_indicators 
                           if indicator in summary or indicator in actions)
        
        # Classify based on evidence strength
        if evidence_count >= 2 and 'ghost probing' in key_actions:
            predictions[video_id] = 'ghost_probing'
        else:
            predictions[video_id] = 'normal_traffic'
    
    return predictions

def calculate_metrics(ground_truth: Dict[str, str], predictions: Dict[str, str]) -> Dict:
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
    """Main analysis function"""
    
    # File paths
    ground_truth_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv"
    videochat2_results_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/videochat/blue_jewel_results/artifacts/outputs"
    
    print("=" * 80)
    print("COMPREHENSIVE VIDEOCHAT2 GHOST PROBING ANALYSIS")
    print("=" * 80)
    
    # Step 1: Analyze VideoChat2's actual behavior
    print("\n1. ANALYZING VIDEOCHAT2 BEHAVIOR PATTERNS...")
    behavior_analysis = analyze_videochat2_behavior(videochat2_results_dir)
    
    print(f"Total files processed: {behavior_analysis['total_files']}")
    print(f"Sentiment distribution: {behavior_analysis['sentiment_distribution']}")
    print(f"Scene theme distribution: {behavior_analysis['scene_theme_distribution']}")
    print(f"Key actions patterns: {behavior_analysis['key_actions_patterns']}")
    
    # Step 2: Load ground truth and filter to 100-video dataset
    print("\n2. LOADING GROUND TRUTH FOR 100-VIDEO DATASET...")
    all_ground_truth = load_ground_truth_labels(ground_truth_path)
    video_list = get_100_video_dataset()
    ground_truth = {vid: all_ground_truth[vid] for vid in video_list if vid in all_ground_truth}
    print(f"Ground truth labels loaded: {len(ground_truth)}")
    
    # Count ground truth distribution
    gt_ghost = sum(1 for label in ground_truth.values() if label == 'ghost_probing')
    gt_normal = sum(1 for label in ground_truth.values() if label == 'normal_traffic')
    print(f"Ground truth distribution: {gt_ghost} ghost probing, {gt_normal} normal traffic")
    
    # Step 3: Apply corrected VideoChat2 classification
    print("\n3. APPLYING CORRECTED VIDEOCHAT2 CLASSIFICATION...")
    all_predictions = corrected_videochat2_classification(videochat2_results_dir)
    predictions = {vid: all_predictions[vid] for vid in video_list if vid in all_predictions}
    
    # Count prediction distribution
    pred_ghost = sum(1 for label in predictions.values() if label == 'ghost_probing')
    pred_normal = sum(1 for label in predictions.values() if label == 'normal_traffic')
    print(f"VideoChat2 predictions: {pred_ghost} ghost probing, {pred_normal} normal traffic")
    print(f"Total predictions: {len(predictions)}")
    
    # Step 4: Calculate performance metrics
    print("\n4. CALCULATING PERFORMANCE METRICS...")
    metrics = calculate_metrics(ground_truth, predictions)
    
    # Step 5: Generate detailed report
    print("\n5. PERFORMANCE RESULTS:")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.3f}")
    print("\nConfusion Matrix:")
    print("                    Predicted")
    print("                Ghost    Normal")
    print(f"Actual Ghost    {metrics['true_positives']:5d}    {metrics['false_negatives']:5d}")
    print(f"       Normal   {metrics['false_positives']:5d}    {metrics['true_negatives']:5d}")
    
    # Step 6: Show videos classified as NOT ghost probing
    print("\n6. VIDEOS CLASSIFIED AS NOT GHOST PROBING:")
    print("=" * 50)
    normal_predictions = [vid for vid, pred in predictions.items() if pred == 'normal_traffic']
    print(f"Total videos classified as normal traffic: {len(normal_predictions)}")
    
    for video_id in sorted(normal_predictions):
        gt_label = ground_truth.get(video_id, 'unknown')
        status = "✓ Correct" if gt_label == 'normal_traffic' else "✗ Missed ghost probing"
        print(f"  {video_id}: Ground Truth = {gt_label:15s} | {status}")
    
    # Step 7: Error analysis
    print("\n7. ERROR ANALYSIS:")
    print("=" * 50)
    
    false_positives = [(vid, gt, pred) for vid, gt, pred in metrics['incorrect_predictions'] 
                      if pred == 'ghost_probing' and gt == 'normal_traffic']
    
    false_negatives = [(vid, gt, pred) for vid, gt, pred in metrics['incorrect_predictions'] 
                      if pred == 'normal_traffic' and gt == 'ghost_probing']
    
    if false_positives:
        print(f"\nFALSE POSITIVES ({len(false_positives)} videos):")
        print("Videos incorrectly classified as ghost probing:")
        for vid, gt, pred in false_positives[:10]:  # Show first 10
            print(f"  {vid}")
        if len(false_positives) > 10:
            print(f"  ... and {len(false_positives) - 10} more")
    
    if false_negatives:
        print(f"\nFALSE NEGATIVES ({len(false_negatives)} videos):")
        print("Ghost probing videos missed by VideoChat2:")
        for vid, gt, pred in false_negatives[:10]:  # Show first 10
            print(f"  {vid}")
        if len(false_negatives) > 10:
            print(f"  ... and {len(false_negatives) - 10} more")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()