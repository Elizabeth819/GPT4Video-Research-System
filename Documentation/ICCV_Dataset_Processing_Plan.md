# ICCV论文数据集重新处理和记录方案

Created: 2025-07-07 16:05:00  
Purpose: 为ICCV论文重新生成完整、有标记的数据集，解决假阳性问题

## 📊 Current Status Assessment

### RUN-003/RUN-005 Status (In Progress)
- **Current Progress**: 94/100 videos completed (94%)
- **Processing Model**: gpt-4o-global (Azure GPT-4o)
- **Prompt Version**: GP3S-V1-ENH (Enhanced 3-step ghost probing detection)
- **API Version**: 2025-04-01-preview
- **Expected Completion**: ~30-60 minutes

### Previous ICCV Data Issue
- **Problem**: Cannot identify which exact dataset was used in ICCV submission
- **Impact**: Reviewers criticized small dataset size (20 videos)
- **Solution**: Generate new, well-documented dataset with proper marking

## 🎯 New ICCV Dataset Processing Plan

### Phase 1: Complete Current RUN-005 (Immediate)
- **Wait for completion** of current 100 videos processing
- **Verify all 100 videos** are successfully processed
- **Generate initial analysis** to assess false positive issues

### Phase 2: Create ICCV-Specific Dataset (New)

#### Dataset Configuration
```bash
# ICCV Paper Dataset - Version 2025.07.07
# Purpose: Address false positive issues and provide comprehensive evaluation

Dataset Name: "ICCV-2025-Enhanced-100v"
Total Videos: 100 (from labels.csv)
Processing Model: gpt-4o-global
Prompt Version: GP3S-V1-ENH-FalsePositiveReduced
API Version: 2025-04-01-preview
Output Directory: result/iccv-2025-enhanced-100v/
```

#### Enhanced Prompt for False Positive Reduction
1. **Stricter Detection Criteria**: More conservative ghost probing identification
2. **Multi-step Validation**: Require multiple evidence points
3. **Context Awareness**: Consider surrounding traffic context
4. **Temporal Consistency**: Verify behavior over multiple frames

### Phase 3: Subset Selection for Paper

#### Create Multiple Dataset Sizes
1. **Full Dataset**: 100 videos (complete evaluation)
2. **High-Quality Subset**: 50 videos (balanced ghost probing/non-events)
3. **Paper Core**: 20 videos (specifically chosen challenging cases)
4. **Validation Set**: 30 videos (for cross-validation)

## 🏷️ Comprehensive Marking and Documentation System

### File Naming Convention
```
# Pattern: actionSummary_{video_id}_{model}_{version}_{date}.json
# Example: actionSummary_images_1_001_gpt4o_enhanced_20250707.json
```

### Metadata Documentation for Each Run
```json
{
  "run_info": {
    "run_id": "ICCV-2025-001",
    "purpose": "ICCV paper dataset generation",
    "date": "2025-07-07",
    "model": "gpt-4o-global",
    "prompt_version": "GP3S-V1-ENH-FPR",
    "video_count": 100,
    "source_videos": "DADA-2000-videos (based on labels.csv)"
  },
  "processing_config": {
    "frame_interval": 10,
    "frames_per_interval": 8,
    "api_version": "2025-04-01-preview",
    "temperature": 0.0,
    "false_positive_reduction": true
  },
  "quality_metrics": {
    "false_positive_rate": "TBD",
    "false_negative_rate": "TBD",
    "accuracy": "TBD",
    "reviewer_feedback_addressed": ["small_dataset", "false_positives"]
  }
}
```

### Paper Documentation Tracking
```markdown
# ICCV Paper Data Usage Tracking

## Dataset Versions Used
1. **Primary Dataset**: ICCV-2025-Enhanced-100v (100 videos)
2. **Paper Figures**: Subset of 20 carefully selected videos
3. **Performance Metrics**: Based on full 100 video analysis
4. **Comparison Data**: Previous baseline vs enhanced results

## Addressing Reviewer Concerns
- ✅ Expanded from 20 to 100 videos
- ✅ Added false positive reduction measures
- ✅ Comprehensive documentation
- ✅ Multiple evaluation subsets
```

## 🔧 Implementation Steps

### Step 1: Wait for Current Completion (16:05-17:00)
```bash
# Monitor current processing
watch "ls result/gpt-4o-100-improved/*.json | wc -l"
```

### Step 2: Analyze Current Results for False Positives
```bash
# Run accuracy analysis on completed dataset
python ghost_probing_accuracy_analysis.py --input result/gpt-4o-100-improved --output analysis_current.json
```

### Step 3: Design Enhanced Prompt for False Positive Reduction
- Review current false positive cases
- Create stricter detection criteria
- Implement multi-step validation process

### Step 4: Execute New ICCV Dataset Processing
```bash
# Create new directory
mkdir -p result/iccv-2025-enhanced-100v

# Process with enhanced prompts
python ActionSummary_o1_o3_batch.py \
  --input_dir DADA-2000-videos \
  --output_dir result/iccv-2025-enhanced-100v \
  --model gpt-4o \
  --num_videos 100 \
  --start_from 0 \
  --frame_interval 10 \
  --frames_per_interval 8 \
  --prompt_version "GP3S-V1-ENH-FPR" \
  --run_id "ICCV-2025-001"
```

### Step 5: Generate Comprehensive Analysis
```bash
# Comparative analysis
python compare_dataset_results.py \
  --baseline result/gpt-4o-100-improved \
  --enhanced result/iccv-2025-enhanced-100v \
  --output iccv_improvement_analysis.md
```

## 📈 Success Metrics

### Technical Metrics
- **False Positive Reduction**: Target <20% (from current 36%)
- **Accuracy Improvement**: Target >70% (from current 54.3%)
- **Precision Improvement**: Target >70% (from current 54.1%)
- **Maintained Recall**: Keep >75% (current 81.6%)

### Paper Metrics
- **Dataset Size**: 100 videos (5x increase from original 20)
- **Documentation Quality**: Complete metadata and tracking
- **Reproducibility**: Full processing pipeline documented
- **Reviewer Concerns**: Addressed false positive and dataset size issues

## 🚨 Critical Success Factors

1. **Complete Documentation**: Every step and decision recorded
2. **Version Control**: Clear versioning of all datasets and models
3. **Quality Validation**: Manual spot-checking of results
4. **Performance Tracking**: Before/after comparisons
5. **Paper Integration**: Clear mapping to paper figures and claims

---

*This plan will be executed immediately upon completion of current RUN-005 processing.*