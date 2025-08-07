# Gemini 1.5 Flash Run12 Performance Analysis Report

## Executive Summary

This report analyzes the performance of Gemini 1.5 Flash model for ghost probing detection in 91 completed videos from the DADA-100 dataset during the run12 experiment.

## Dataset Overview

- **Total Ground Truth Videos**: 100
- **Successfully Processed Videos**: 91 
- **Missing Videos**: 9 (failed processing)
- **Ghost Probing Events in Ground Truth**: 45 out of 91 processed videos (49.5%)

## Performance Metrics

### Confusion Matrix
| Metric | Count |
|--------|-------|
| True Positives (TP) | 15 |
| False Positives (FP) | 17 |
| True Negatives (TN) | 28 |
| False Negatives (FN) | 30 |
| **Total Videos** | **90** |

### Key Performance Indicators
| Metric | Value | Formula |
|--------|-------|---------|
| **Precision** | 46.88% | TP/(TP+FP) = 15/32 |
| **Recall** | 33.33% | TP/(TP+FN) = 15/45 |
| **F1-Score** | 38.96% | 2×(Precision×Recall)/(Precision+Recall) |
| **Specificity** | 62.22% | TN/(TN+FP) = 28/45 |
| **Accuracy** | 47.78% | (TP+TN)/(TP+TN+FP+FN) = 43/90 |

## Analysis Findings

### Detection Performance
- **True Positive Rate**: Only 33.33% of actual ghost probing events were correctly identified
- **False Positive Rate**: 37.78% of negative cases were incorrectly flagged as ghost probing
- **Overall Accuracy**: Below 50%, indicating poor discriminative ability

### Pattern Analysis

#### True Positives (15 videos correctly identified)
Most successful detections occurred in videos from categories 1, 4, and 5:
- images_1_003, images_1_006, images_1_010, images_1_011, images_1_013, images_1_014, images_1_016
- images_4_002, images_4_003, images_4_006
- images_5_012, images_5_020, images_5_027, images_5_031, images_5_037

#### False Negatives (30 videos missed)
High miss rate across all categories, particularly in category 5:
- Category 1: 10 missed events
- Category 3: 4 missed events  
- Category 4: 2 missed events
- Category 5: 14 missed events

#### False Positives (17 videos incorrectly flagged)
Significant over-detection across categories:
- Category 1: 4 false alarms
- Category 2: 1 false alarm
- Category 4: 2 false alarms
- Category 5: 10 false alarms

## Missing Data Impact

9 videos were not processed, including 8 videos with ground truth ghost probing events:
- images_5_011.avi: 3s ghost probing
- images_5_046.avi: 2s ghost probing
- images_5_047.avi: 2s ghost probing
- images_5_049.avi: 3s ghost probing
- images_5_050.avi: 2s ghost probing
- images_5_051.avi: 2s ghost probing
- images_5_052.avi: 3s ghost probing
- images_5_053.avi: 7s ghost probing

## Key Observations

### Model Strengths
1. **Some Detection Capability**: Successfully identified 15 ghost probing events
2. **Temporal Accuracy**: When detected, events were typically identified in the correct time segments
3. **Category Performance**: Better performance in categories 1 and 4 compared to others

### Model Weaknesses
1. **High Miss Rate**: 66.67% of actual ghost probing events were missed
2. **High False Positive Rate**: 37.78% false alarm rate
3. **Inconsistent Performance**: Large variation in detection across video categories
4. **Poor Discriminative Ability**: Below-chance accuracy suggests difficulty distinguishing ghost probing from normal driving scenarios

### Technical Issues
1. **Processing Failures**: 9% of videos failed to process completely
2. **Category 5 Challenges**: Particularly poor performance in category 5 videos (high miss rate and false positives)

## Recommendations

### Immediate Improvements
1. **Threshold Adjustment**: Consider adjusting detection sensitivity to reduce false positives
2. **Category-Specific Tuning**: Develop specialized detection parameters for different video categories
3. **Processing Stability**: Address technical issues causing processing failures

### Model Enhancement
1. **Training Data**: Increase training examples, particularly for category 5 scenarios
2. **Feature Engineering**: Improve temporal sequence analysis for ghost probing detection
3. **Context Understanding**: Better integration of driving context and scene understanding

### Evaluation Protocol
1. **Expanded Dataset**: Process all 100 videos to eliminate missing data bias
2. **Temporal Analysis**: Implement time-based matching tolerance for ground truth comparison
3. **Qualitative Analysis**: Manual review of false positives/negatives to identify pattern improvements

## Conclusion

The Gemini 1.5 Flash model shows limited effectiveness for ghost probing detection in its current configuration, with performance significantly below acceptable thresholds for safety-critical applications. The 47.78% accuracy and 33.33% recall indicate substantial room for improvement in both detection sensitivity and specificity. The high false positive rate (37.78%) is particularly concerning for practical deployment scenarios where false alarms could impact system reliability.

The model demonstrates some capability to identify ghost probing events when they occur but lacks the precision and consistency required for robust autonomous driving safety systems.