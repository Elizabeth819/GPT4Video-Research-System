"""
Evaluation script for ghost probing detection performance.
Compares model predictions with ground truth labels and calculates metrics.
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GhostProbingEvaluator:
    """Evaluates ghost probing detection performance against ground truth."""
    
    def __init__(self, ground_truth_path: str):
        """
        Initialize evaluator with ground truth data.
        
        Args:
            ground_truth_path: Path to ground truth CSV file
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth = self._load_ground_truth()
        
    def _load_ground_truth(self) -> Dict[str, Dict]:
        """
        Load and parse ground truth labels.
        
        Returns:
            Dictionary mapping video IDs to ground truth information
        """
        ground_truth = {}
        
        try:
            # Read ground truth CSV
            df = pd.read_csv(self.ground_truth_path, sep='\t', header=None, 
                           names=['video_id', 'ground_truth_label'])
            
            logger.info(f"Loaded {len(df)} ground truth entries")
            
            for _, row in df.iterrows():
                video_id = row['video_id']
                label = row['ground_truth_label']
                
                if pd.isna(video_id) or pd.isna(label):
                    continue
                    
                # Parse label
                parsed_label = self._parse_ground_truth_label(label)
                ground_truth[video_id] = parsed_label
                
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            raise
            
        logger.info(f"Parsed ground truth for {len(ground_truth)} videos")
        return ground_truth
    
    def _parse_ground_truth_label(self, label: str) -> Dict:
        """
        Parse ground truth label to extract ghost probing information.
        
        Args:
            label: Ground truth label string
            
        Returns:
            Parsed label information
        """
        label_lower = str(label).lower()
        
        # Extract timestamp if present
        timestamp_match = re.search(r'(\d+)s?:', label_lower)
        timestamp = int(timestamp_match.group(1)) if timestamp_match else None
        
        # Determine if ghost probing is present
        is_ghost_probing = 'ghost probing' in label_lower
        is_cut_in = 'cut-in' in label_lower or 'cutin' in label_lower
        
        return {
            'original_label': label,
            'is_ghost_probing': is_ghost_probing,
            'is_cut_in': is_cut_in,
            'timestamp': timestamp,
            'binary_label': 1 if is_ghost_probing else 0
        }
    
    def _extract_video_id_from_path(self, video_path: str) -> str:
        """
        Extract standardized video ID from file path.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Standardized video ID
        """
        filename = os.path.basename(video_path)
        # Remove extension
        video_id = os.path.splitext(filename)[0]
        
        # Handle different naming conventions
        if video_id.endswith('.avi'):
            video_id = video_id[:-4]
            
        return filename  # Use full filename as key
    
    def _parse_prediction_result(self, result: Dict[str, Any]) -> Dict:
        """
        Parse prediction result to extract ghost probing detection.
        
        Args:
            result: Analysis result from ghost probing detector
            
        Returns:
            Parsed prediction information
        """
        if 'error' in result:
            return {
                'prediction': 0,
                'confidence': 'error',
                'has_high_confidence': False,
                'has_potential': False,
                'analysis_text': result.get('error', '')
            }
        
        ghost_probing_info = result.get('ghost_probing_detected', {})
        analysis_text = result.get('analysis', '')
        
        # Determine binary prediction
        high_confidence = ghost_probing_info.get('high_confidence_ghost_probing', False)
        potential = ghost_probing_info.get('potential_ghost_probing', False)
        any_detection = ghost_probing_info.get('any_ghost_probing', False)
        
        return {
            'prediction': 1 if any_detection else 0,
            'confidence': ghost_probing_info.get('confidence_level', 'none'),
            'has_high_confidence': high_confidence,
            'has_potential': potential,
            'analysis_text': analysis_text
        }
    
    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate prediction results against ground truth.
        
        Args:
            results: List of prediction results from ghost probing detector
            
        Returns:
            Evaluation metrics and detailed analysis
        """
        y_true = []
        y_pred = []
        detailed_results = []
        
        matched_videos = 0
        unmatched_videos = []
        
        for result in results:
            video_path = result.get('video_path', '')
            video_id = self._extract_video_id_from_path(video_path)
            
            # Find ground truth for this video
            if video_id not in self.ground_truth:
                unmatched_videos.append(video_id)
                continue
                
            ground_truth = self.ground_truth[video_id]
            prediction = self._parse_prediction_result(result)
            
            y_true.append(ground_truth['binary_label'])
            y_pred.append(prediction['prediction'])
            
            detailed_results.append({
                'video_id': video_id,
                'video_path': video_path,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': ground_truth['binary_label'] == prediction['prediction']
            })
            
            matched_videos += 1
        
        if not y_true:
            raise ValueError("No matching videos found between predictions and ground truth")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Additional analysis
        correct_predictions = sum(1 for r in detailed_results if r['correct'])
        
        evaluation_results = {
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            },
            'summary': {
                'total_predictions': len(results),
                'matched_videos': matched_videos,
                'unmatched_videos': len(unmatched_videos),
                'correct_predictions': correct_predictions,
                'accuracy_percentage': accuracy * 100
            },
            'detailed_results': detailed_results,
            'unmatched_videos': unmatched_videos
        }
        
        return evaluation_results
    
    def generate_report(self, evaluation_results: Dict[str, Any], output_path: str = None):
        """
        Generate detailed evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_results
            output_path: Optional path to save report
        """
        metrics = evaluation_results['metrics']
        summary = evaluation_results['summary']
        
        report = f"""
# Ghost Probing Detection Evaluation Report

## Summary
- Total videos processed: {summary['total_predictions']}
- Successfully matched with ground truth: {summary['matched_videos']}
- Unmatched videos: {summary['unmatched_videos']}
- Overall accuracy: {summary['accuracy_percentage']:.2f}%

## Performance Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1 Score**: {metrics['f1_score']:.4f}

## Confusion Matrix
```
                Predicted
                No    Yes
Actual No    {metrics['true_negatives']:4d}  {metrics['false_positives']:4d}
Actual Yes   {metrics['false_negatives']:4d}  {metrics['true_positives']:4d}
```

## Detailed Analysis
- True Positives (Correct ghost probing detection): {metrics['true_positives']}
- False Positives (Incorrect ghost probing detection): {metrics['false_positives']}
- True Negatives (Correct no ghost probing): {metrics['true_negatives']}
- False Negatives (Missed ghost probing): {metrics['false_negatives']}

## Error Analysis
"""
        
        # Add false positive analysis
        false_positives = [r for r in evaluation_results['detailed_results'] 
                          if not r['correct'] and r['prediction']['prediction'] == 1]
        
        if false_positives:
            report += f"\n### False Positives ({len(false_positives)} cases):\n"
            for fp in false_positives[:5]:  # Show first 5
                report += f"- {fp['video_id']}: Predicted ghost probing but ground truth is '{fp['ground_truth']['original_label']}'\n"
        
        # Add false negative analysis
        false_negatives = [r for r in evaluation_results['detailed_results'] 
                          if not r['correct'] and r['prediction']['prediction'] == 0]
        
        if false_negatives:
            report += f"\n### False Negatives ({len(false_negatives)} cases):\n"
            for fn in false_negatives[:5]:  # Show first 5
                report += f"- {fn['video_id']}: Missed ghost probing at '{fn['ground_truth']['original_label']}'\n"
        
        # Add unmatched videos
        if evaluation_results['unmatched_videos']:
            report += f"\n### Unmatched Videos ({len(evaluation_results['unmatched_videos'])} cases):\n"
            for video in evaluation_results['unmatched_videos'][:10]:  # Show first 10
                report += f"- {video}\n"
        
        print(report)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"Report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
    
    def save_detailed_results(self, evaluation_results: Dict[str, Any], output_path: str):
        """
        Save detailed evaluation results to JSON file.
        
        Args:
            evaluation_results: Results from evaluate_results
            output_path: Path to save detailed results
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Detailed results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving detailed results: {e}")


def main():
    """Test the evaluator with sample data."""
    ground_truth_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv"
    
    if not os.path.exists(ground_truth_path):
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        return
    
    # Initialize evaluator
    evaluator = GhostProbingEvaluator(ground_truth_path)
    
    print(f"Loaded ground truth for {len(evaluator.ground_truth)} videos")
    
    # Show some ground truth samples
    print("\nSample ground truth entries:")
    for i, (video_id, gt) in enumerate(list(evaluator.ground_truth.items())[:5]):
        print(f"{video_id}: {gt['original_label']} -> {gt['binary_label']}")


if __name__ == "__main__":
    main()