# DADA-100 Subset Usage Information

## Dataset Source
- **Base Dataset**: DADA-2000 (Fang et al. 2019)
- **Public Availability**: https://github.com/JWFangit/LOTVS-DADA
- **License**: Open for research use
- **Citation**: Fang, J.; Yan, D.; Qiao, J.; Xue, J.; and Wang, H. 2019. DADA-2000: Can Driving Accident be Predicted by Driver Attention? Analyzed by A Benchmark.

## DADA-100 Subset Selection
Our experiments use a curated subset of 100 videos from the DADA-2000 dataset, selected based on:
- Video quality and clarity
- Presence of safety-critical events (ghost probing, cut-in)
- Balanced representation of different scenarios
- Sufficient temporal context for analysis

## Ground Truth Annotations
We created manual annotations for ghost probing detection on this subset:
- **File**: `groundtruth_labels.csv` (included in supplementary materials)
- **Format**: `video_id,ground_truth_label,timestamp,notes`
- **Labels**: Ghost probing events with precise temporal annotations
- **Total Events**: 54 ghost probing events, 47 normal scenarios

## Video Processing Configuration
- **Frame Interval**: 10 seconds
- **Frames Per Interval**: 10 frames (1 FPS)
- **Resolution**: Original 1584×660 maintained
- **Audio**: Full track extraction with Whisper transcription

## Reproducibility Notes
Researchers can reproduce our results by:
1. Downloading DADA-2000 from the official source
2. Using the video IDs provided in `video_ids_used.txt`
3. Applying our processing pipeline with the provided scripts
4. Comparing against our ground truth annotations

## Contact
For questions about the DADA-100 subset selection or annotations, please refer to the main paper methodology section.