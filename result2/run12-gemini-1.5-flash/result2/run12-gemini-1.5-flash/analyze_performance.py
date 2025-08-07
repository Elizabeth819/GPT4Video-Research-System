#!/usr/bin/env python3
"""
Performance analysis script for Gemini run12 experiment
Calculates precision, recall, F1-score, specificity, and accuracy
for ghost probing detection against ground truth labels.
"""

import json
import os
import csv
from collections import defaultdict

def load_ground_truth(csv_path):
    """Load ground truth labels from CSV file"""
    ground_truth = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            video_id = row['video_id']
            label = row['ground_truth_label']
            # Convert to boolean: True if contains "ghost probing", False otherwise
            has_ghost_probing = "ghost probing" in label.lower()
            ground_truth[video_id] = has_ghost_probing
    return ground_truth

def analyze_json_file(json_path):
    """Analyze a single JSON file for ghost probing predictions"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract video_id from filename
        filename = os.path.basename(json_path)
        video_id = filename.replace('actionSummary_', '').replace('.json', '') + '.avi'
        
        # Check all segments for ghost probing in key_actions
        has_ghost_probing = False
        ghost_probing_segments = []
        
        for segment in data:
            if isinstance(segment, dict) and 'key_actions' in segment:
                key_actions = segment.get('key_actions', '')
                if key_actions and 'ghost probing' in key_actions.lower():
                    has_ghost_probing = True
                    ghost_probing_segments.append({
                        'segment_id': segment.get('segment_id', 'unknown'),
                        'timestamp': f"{segment.get('Start_Timestamp', 'unknown')}-{segment.get('End_Timestamp', 'unknown')}",
                        'key_actions': key_actions
                    })
        
        return video_id, has_ghost_probing, ghost_probing_segments
    
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None, False, []

def calculate_metrics(tp, fp, tn, fn):
    """Calculate performance metrics"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'accuracy': accuracy
    }

def main():
    # Paths
    ground_truth_path = '/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv'
    results_dir = '/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run12-gemini-1.5-flash/result2/run12-gemini-1.5-flash/'
    
    # Load ground truth
    print("Loading ground truth labels...")
    ground_truth = load_ground_truth(ground_truth_path)
    print(f"Loaded {len(ground_truth)} ground truth labels")
    
    # Process all JSON files
    print("\nProcessing JSON result files...")
    gemini_predictions = {}
    detailed_results = []
    
    json_files = [f for f in os.listdir(results_dir) if f.startswith('actionSummary_') and f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in sorted(json_files):
        json_path = os.path.join(results_dir, json_file)
        video_id, has_ghost_probing, ghost_segments = analyze_json_file(json_path)
        
        if video_id:
            gemini_predictions[video_id] = has_ghost_probing
            detailed_results.append({
                'video_id': video_id,
                'gemini_prediction': has_ghost_probing,
                'ghost_segments': ghost_segments
            })
    
    print(f"Processed {len(gemini_predictions)} videos")
    
    # Calculate confusion matrix
    tp = fp = tn = fn = 0
    classification_details = []
    
    for video_id in ground_truth:
        if video_id in gemini_predictions:
            gt_label = ground_truth[video_id]
            gemini_pred = gemini_predictions[video_id]
            
            if gt_label and gemini_pred:
                tp += 1
                result_type = "True Positive"
            elif not gt_label and not gemini_pred:
                tn += 1
                result_type = "True Negative"
            elif not gt_label and gemini_pred:
                fp += 1
                result_type = "False Positive"
            elif gt_label and not gemini_pred:
                fn += 1
                result_type = "False Negative"
            
            # Find detailed results for this video
            video_details = next((r for r in detailed_results if r['video_id'] == video_id), None)
            ghost_segments = video_details['ghost_segments'] if video_details else []
            
            classification_details.append({
                'video_id': video_id,
                'ground_truth': gt_label,
                'gemini_prediction': gemini_pred,
                'result_type': result_type,
                'ghost_segments': ghost_segments
            })
    
    # Calculate metrics
    metrics = calculate_metrics(tp, fp, tn, fn)
    
    # Print results
    print("\n" + "="*80)
    print("GEMINI RUN12 PERFORMANCE ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nConfusion Matrix:")
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"Total Videos:         {tp + fp + tn + fn}")
    
    print(f"\nPerformance Metrics:")
    print(f"Precision:    {metrics['precision']:.4f} ({tp}/{tp + fp})")
    print(f"Recall:       {metrics['recall']:.4f} ({tp}/{tp + fn})")
    print(f"F1-Score:     {metrics['f1_score']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f} ({tn}/{tn + fp})")
    print(f"Accuracy:     {metrics['accuracy']:.4f} ({tp + tn}/{tp + tn + fp + fn})")
    
    # Detailed breakdown
    print(f"\n" + "="*80)
    print("DETAILED CLASSIFICATION BREAKDOWN")
    print("="*80)
    
    # Group by result type
    for result_type in ["True Positive", "False Positive", "True Negative", "False Negative"]:
        videos_of_type = [d for d in classification_details if d['result_type'] == result_type]
        if videos_of_type:
            print(f"\n{result_type} ({len(videos_of_type)} videos):")
            for video in sorted(videos_of_type, key=lambda x: x['video_id']):
                print(f"  {video['video_id']}: GT={video['ground_truth']}, Gemini={video['gemini_prediction']}")
                if video['ghost_segments']:
                    for segment in video['ghost_segments']:
                        print(f"    - {segment['segment_id']} ({segment['timestamp']}): {segment['key_actions']}")
    
    # Summary by ground truth
    print(f"\n" + "="*80)
    print("SUMMARY BY GROUND TRUTH LABEL")
    print("="*80)
    
    gt_positive = [d for d in classification_details if d['ground_truth']]
    gt_negative = [d for d in classification_details if not d['ground_truth']]
    
    print(f"\nVideos with Ghost Probing in Ground Truth ({len(gt_positive)} videos):")
    for video in sorted(gt_positive, key=lambda x: x['video_id']):
        status = "✓ DETECTED" if video['gemini_prediction'] else "✗ MISSED"
        print(f"  {video['video_id']}: {status}")
    
    print(f"\nVideos without Ghost Probing in Ground Truth ({len(gt_negative)} videos):")
    for video in sorted(gt_negative, key=lambda x: x['video_id']):
        status = "✗ FALSE ALARM" if video['gemini_prediction'] else "✓ CORRECT"
        print(f"  {video['video_id']}: {status}")
    
    # Pattern analysis
    print(f"\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    print(f"\nFalse Positives Analysis:")
    fp_videos = [d for d in classification_details if d['result_type'] == "False Positive"]
    if fp_videos:
        print(f"Gemini incorrectly identified ghost probing in {len(fp_videos)} videos:")
        for video in sorted(fp_videos, key=lambda x: x['video_id']):
            print(f"  {video['video_id']}:")
            for segment in video['ghost_segments']:
                print(f"    - {segment['segment_id']} ({segment['timestamp']}): {segment['key_actions']}")
    else:
        print("No false positives detected!")
    
    print(f"\nFalse Negatives Analysis:")
    fn_videos = [d for d in classification_details if d['result_type'] == "False Negative"]
    if fn_videos:
        print(f"Gemini missed ghost probing in {len(fn_videos)} videos:")
        for video in sorted(fn_videos, key=lambda x: x['video_id']):
            print(f"  {video['video_id']}: Ground truth has ghost probing but Gemini detected none")
    else:
        print("No false negatives detected!")

if __name__ == "__main__":
    main()