#!/usr/bin/env python3
"""
Metric Computation Tools for AutoDrive-GPT Evaluation
Implements F1-score, precision, recall, and accuracy calculations
as described in Section 4.2 of the paper.
"""

import json
import csv
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
import scipy.stats as stats

def load_ground_truth(csv_file: str) -> Dict[str, str]:
    """
    Load ground truth labels from CSV file.
    Format: video_id,ground_truth_label,notes
    """
    ground_truth = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video_id'].replace('.mp4', '').replace('.avi', '')
            ground_truth[video_id] = row['ground_truth_label']
    return ground_truth

def parse_prediction(prediction: str) -> bool:
    """
    Parse model prediction to binary ghost probing classification.
    Returns True if ghost probing detected, False otherwise.
    """
    if not prediction or prediction.lower() == 'none':
        return False
    
    # Check for ghost probing keywords
    ghost_keywords = ['ghost probing', 'ghost_probing', 'ghosting', 'sudden emergence']
    prediction_lower = prediction.lower()
    
    return any(keyword in prediction_lower for keyword in ghost_keywords)

def parse_ground_truth(label: str) -> bool:
    """
    Parse ground truth label to binary classification.
    Returns True if ghost probing event, False otherwise.
    """
    if not label or label.lower() == 'none':
        return False
    
    return 'ghost probing' in label.lower()

def calculate_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: List of ground truth binary labels
        y_pred: List of predicted binary labels
    
    Returns:
        Dictionary containing precision, recall, f1_score, and accuracy
    """
    # Convert to numpy arrays for sklearn
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    
    # Calculate metrics
    precision = precision_score(y_true_arr, y_pred_arr, zero_division=0.0)
    recall = recall_score(y_true_arr, y_pred_arr, zero_division=0.0)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0.0)
    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'confusion_matrix': cm.tolist()
    }

def evaluate_model_predictions(predictions_file: str, ground_truth_file: str) -> Dict[str, float]:
    """
    Evaluate model predictions against ground truth.
    
    Args:
        predictions_file: JSON file containing model predictions
        ground_truth_file: CSV file containing ground truth labels
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_file)
    
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    y_true = []
    y_pred = []
    matched_videos = []
    
    # Match predictions with ground truth
    for video_id, prediction_data in predictions.items():
        if video_id in ground_truth:
            # Extract key_actions from prediction
            key_actions = prediction_data.get('key_actions', '')
            
            # Parse labels
            true_label = parse_ground_truth(ground_truth[video_id])
            pred_label = parse_prediction(key_actions)
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            matched_videos.append(video_id)
    
    print(f"Matched {len(matched_videos)} videos for evaluation")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    metrics['num_videos'] = len(matched_videos)
    metrics['matched_videos'] = matched_videos
    
    return metrics

def paired_t_test(model1_results: List[float], model2_results: List[float]) -> Dict[str, float]:
    """
    Perform paired t-test for statistical significance testing.
    
    Args:
        model1_results: List of F1-scores for model 1
        model2_results: List of F1-scores for model 2
    
    Returns:
        Dictionary containing t-statistic, p-value, degrees of freedom, and Cohen's d
    """
    # Ensure equal length
    min_len = min(len(model1_results), len(model2_results))
    model1_scores = np.array(model1_results[:min_len])
    model2_scores = np.array(model2_results[:min_len])
    
    # Calculate differences
    differences = model1_scores - model2_scores
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    
    # Calculate Cohen's d
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff != 0 else 0.0
    
    # Calculate 95% confidence interval
    se_diff = std_diff / np.sqrt(len(differences))
    ci_95 = stats.t.interval(0.95, len(differences)-1, mean_diff, se_diff)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'degrees_freedom': len(differences) - 1,
        'cohens_d': cohens_d,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'confidence_interval_95': ci_95,
        'num_pairs': len(differences)
    }

def compare_models(model1_file: str, model2_file: str, ground_truth_file: str) -> Dict:
    """
    Compare two models and perform statistical analysis.
    
    Args:
        model1_file: JSON file with model 1 predictions
        model2_file: JSON file with model 2 predictions  
        ground_truth_file: CSV file with ground truth labels
    
    Returns:
        Dictionary containing comparison results
    """
    # Evaluate both models
    model1_metrics = evaluate_model_predictions(model1_file, ground_truth_file)
    model2_metrics = evaluate_model_predictions(model2_file, ground_truth_file)
    
    # Find common videos
    common_videos = set(model1_metrics['matched_videos']) & set(model2_metrics['matched_videos'])
    print(f"Common videos for statistical testing: {len(common_videos)}")
    
    # For statistical testing, we would need individual video F1-scores
    # This is a simplified version - in practice, you'd calculate per-video metrics
    
    return {
        'model1_metrics': model1_metrics,
        'model2_metrics': model2_metrics,
        'common_videos_count': len(common_videos),
        'f1_improvement': model1_metrics['f1_score'] - model2_metrics['f1_score'],
        'recall_improvement': model1_metrics['recall'] - model2_metrics['recall'],
        'precision_improvement': model1_metrics['precision'] - model2_metrics['precision']
    }

def print_evaluation_results(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print formatted evaluation results.
    """
    print(f"\n{model_name} Evaluation Results:")
    print("=" * 50)
    print(f"F1-Score:     {metrics['f1_score']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Videos:       {metrics['num_videos']}")
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Negatives:  {metrics['true_negatives']}")
    print(f"False Negatives: {metrics['false_negatives']}")

if __name__ == "__main__":
    # Example usage
    print("AutoDrive-GPT Metric Computation Tools")
    print("Usage: python metric_computation.py")
    print("Modify the file paths below for your specific evaluation")
    
    # Example file paths (modify as needed)
    # gpt4o_predictions = "path/to/gpt4o_predictions.json"
    # gemini_predictions = "path/to/gemini_predictions.json"
    # ground_truth_labels = "path/to/ground_truth.csv"
    
    # # Evaluate individual models
    # gpt4o_metrics = evaluate_model_predictions(gpt4o_predictions, ground_truth_labels)
    # print_evaluation_results(gpt4o_metrics, "AutoDrive-GPT (GPT-4o)")
    
    # gemini_metrics = evaluate_model_predictions(gemini_predictions, ground_truth_labels)
    # print_evaluation_results(gemini_metrics, "Gemini 2.0 Flash")
    
    # # Compare models
    # comparison = compare_models(gpt4o_predictions, gemini_predictions, ground_truth_labels)
    # print(f"\nModel Comparison:")
    # print(f"F1-Score Improvement: {comparison['f1_improvement']:.4f}")
    # print(f"Recall Improvement: {comparison['recall_improvement']:.4f}")