# Ghost Probing Detection Accuracy Analysis Report

Generated on: 2025-07-07 16:13:29

## Executive Summary

This report analyzes the accuracy of GPT-4o model's ghost probing detection compared to ground truth labels.

## Dataset Overview

- **Ground Truth Labels**: 100 videos
- **Model Results**: 98 videos
- **Compared Videos**: 97 videos

## Performance Metrics


| Metric | Value |
|--------|-------|
| **Accuracy** | 0.526 (52.6%) |
| **Precision** | 0.536 (53.6%) |
| **Recall** | 0.865 (86.5%) |
| **Specificity** | 0.133 (13.3%) |
| **F1 Score** | 0.662 (66.2%) |

## Confusion Matrix

|                    | Predicted: Ghost Probing | Predicted: No Ghost Probing |
|--------------------|--------------------------|------------------------------|
| **Actual: Ghost Probing** | 45 (True Positives) | 7 (False Negatives) |
| **Actual: No Ghost Probing** | 39 (False Positives) | 6 (True Negatives) |

## Detailed Results

### True Positives (45 videos)
Model correctly detected ghost probing:

- **images_1_011**: Ground truth: "11s: ghost probing"
- **images_5_046**: Ground truth: "2s: ghost probing"
- **images_5_047**: Ground truth: "2s: ghost probing"
- **images_3_007**: Ground truth: "3s: ghost probing"
- **images_5_044**: Ground truth: "4s: ghost probing"
- **images_5_012**: Ground truth: "13s: ghost probing"
- **images_5_020**: Ground truth: "11s: ghost probing"
- **images_5_031**: Ground truth: "2s: ghost probing"
- **images_1_017**: Ground truth: "17s: ghost probing"
- **images_1_014**: Ground truth: "5s: ghost probing"
- ... and 35 more

### False Positives (39 videos)
Model incorrectly detected ghost probing:

- **images_4_007**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_038**: Ground truth: "13: cut-in" | Model predicted: Ghost Probing
- **images_5_019**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_034**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_4_001**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_1_004**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_009**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_001**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_4_008**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_008**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_2_004**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_2_003**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_029**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_022**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_026**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_014**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_1_018**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_048**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_018**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_042**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_040**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_1_019**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_3_005**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_3_001**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_002**: Ground truth: "cut-in" | Model predicted: Ghost Probing
- **images_1_026**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_013**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_2_002**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_016**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_3_004**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_004**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_1_020**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_024**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_032**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_017**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_030**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_1_025**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_1_009**: Ground truth: "none" | Model predicted: Ghost Probing
- **images_5_036**: Ground truth: "none" | Model predicted: Ghost Probing

### False Negatives (7 videos)
Model missed actual ghost probing:

- **images_1_008**: Ground truth: "3s: ghost probing" | Model predicted: No Ghost Probing
- **images_5_011**: Ground truth: "3s: ghost probing" | Model predicted: No Ghost Probing
- **images_1_015**: Ground truth: "5s: ghost probing" | Model predicted: No Ghost Probing
- **images_5_049**: Ground truth: "3s: ghost probing" | Model predicted: No Ghost Probing
- **images_5_015**: Ground truth: "2s: ghost probing" | Model predicted: No Ghost Probing
- **images_5_041**: Ground truth: "5s: ghost probing" | Model predicted: No Ghost Probing
- **images_5_006**: Ground truth: "6s: ghost probing" | Model predicted: No Ghost Probing

### True Negatives (6 videos)
Model correctly identified no ghost probing:

- **images_5_025**: Ground truth: "none"
- **images_2_001**: Ground truth: "none"
- **images_5_003**: Ground truth: "none"
- **images_1_024**: Ground truth: "none"
- **images_1_023**: Ground truth: "none"
- **images_5_007**: Ground truth: "none"

## Data Coverage Issues

### Videos with Missing Ground Truth (1 videos)
- **images_5_039**: Model result: True

### Videos with Missing Model Results (3 videos)
- **images_5_045**: Ground truth: "5s: ghost probing"
- **images_2_005**: Ground truth: "none"
- **images_5_054**: Ground truth: "none"

## Analysis Insights

### Ghost Probing Distribution in Ground Truth

- **Videos with Ghost Probing**: 53 (53.0%)
- **Videos without Ghost Probing**: 47 (47.0%)

### Model Performance Analysis

- **Low Precision**: The model has high false positive rate, meaning it often incorrectly identifies ghost probing.
- **High Recall**: The model successfully identifies most actual ghost probing incidents.
- **Good Overall Performance**: Reasonable balance between precision and recall.

## Recommendations

1. **Review False Positives**: Analyze the 39 false positive cases to understand what patterns the model incorrectly identifies as ghost probing.

2. **Review False Negatives**: Examine the 7 false negative cases to understand what ghost probing patterns the model misses.

3. **Improve Training Data**: Use insights from misclassified cases to improve model training or prompts.

4. **Threshold Tuning**: Consider adjusting detection thresholds if the model outputs confidence scores.

## Technical Details

- **Results Directory**: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/gpt-4o-100-improved
- **Labels File**: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/labels.csv
- **Analysis Script**: ghost_probing_accuracy_analysis.py
- **Report Generated**: 2025-07-07 16:13:29
