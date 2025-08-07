# AutoDrive-GPT Supplementary Materials

This directory contains the supplementary materials for the AAAI26 paper "AutoDrive-GPT: Enhancing Autonomous Driving Behavior Annotation and Prediction Using GPT-4o Prompt Tuning".

## Directory Structure

```
supplementary_material/
├── README.md                     # This file
├── code/                         # Source code implementations
│   ├── ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper.py  # Core video analysis script
│   ├── SECURITY_NOTES.md         # Security considerations and notes
│   ├── cobra_preprocessing.py    # Cobra preprocessing pipeline
│   ├── ingest_scenes.py         # Azure Cognitive Search integration
│   ├── player.jsx               # COBRA video player component
│   ├── route.js                 # COBRA API endpoints
│   └── video_utilities.py       # Video processing utilities
├── data/                        # Dataset files and annotations
│   ├── DADA100_usage_info.md    # DADA-100 dataset usage documentation
│   ├── bilibili_dataset_info.csv # Bilibili dataset video information
│   ├── bilibili_labels.csv      # Bilibili dataset ground truth labels
│   ├── frames_extraction_with_titles/  # Extracted frames from bilibili videos
│   │   ├── bilibili_cutin_*.jpg  # Cut-in scenario frames
│   │   └── bilibili_ghosting_*.jpg # Ghost probing scenario frames
│   ├── groundtruth_labels.csv   # Ground truth labels for evaluation
│   └── video_ids_used.txt       # List of video IDs used in experiments
├── documentation/               # System documentation
│   └── system_requirements.md   # Computing infrastructure specs
└── evaluation/                  # Evaluation scripts and tools
    ├── baseline_no_fewshot.py   # Baseline without few-shot learning
    ├── dada100_ablation_experiment.py  # DADA-100 ablation experiments
    ├── final_100_video_evaluation.py   # DADA-100 comprehensive evaluation
    ├── gemini_prompt_improvement_fixed.py  # Gemini baseline implementation
    ├── metric_computation.py     # Metric calculation tools
    └── statistical_ttest_analysis.py    # Statistical significance testing
```

## Code Files (Section 4.3 & 4.6 Requirements)

### Core Video Processing (`code/`)
- **`ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper.py`**: Complete Cobra preprocessing pipeline with GPT-4o integration (used in run8-rerun experiments)
- **`video_utilities.py`**: Common video processing utilities and helper functions  
- **`ingest_scenes.py`**: Azure Cognitive Search integration for semantic video search

### COBRA Frontend (`code/`)
- **`route.js`**: API endpoints for chat and cognitive search functionality
- **`player.jsx`**: Interactive video player component with analysis overlay
- **`SECURITY_NOTES.md`**: Security considerations and implementation notes

**Note on COBRA Project Completeness:**
COBRA is a comprehensive full-stack application with both frontend (Next.js/React) and backend (Python) components. Due to space constraints in supplementary materials, only the most critical files for reproducibility are included here. The complete COBRA codebase contains additional components including:
- Complete Next.js frontend application structure
- Additional React components and utilities
- Backend API integrations and middleware
- Configuration files and deployment scripts
- Additional utility modules and helper functions

Upon paper acceptance, the authors will make the complete GitHub repository publicly available, providing full access to the entire COBRA system codebase.

All source code files include:
- Comprehensive inline comments with specific paper section references
- Function-level documentation linking implementation to methodology
- Cross-references to paper sections for each major algorithm step
- Implementation notes referencing Section 3 (System Architecture) and Section 4 (Methodology)
- Detailed docstrings explaining the relationship between code and paper contributions

## Evaluation Scripts (Section 4.4 Requirements)

### Model Evaluation (`evaluation/`)
- **`final_100_video_evaluation.py`**: Comprehensive evaluation on DADA-100 dataset
- **`gemini_prompt_improvement_fixed.py`**: Gemini 2.0 Flash baseline implementation  
- **`metric_computation.py`**: F1-score, precision, recall calculation tools

### Run8-Rerun Experiment Analysis (`evaluation/`)
- **`dada100_ablation_experiment.py`**: DADA-100 dataset ablation experiment analysis (used in run8-rerun)
- **`statistical_ttest_analysis.py`**: Statistical significance testing with paired t-test
- **`baseline_no_fewshot.py`**: Baseline experiment without few-shot learning

### Statistical Analysis Tools
- Paired t-test implementation for significance testing
- Confusion matrix analysis
- Cohen's d effect size calculation
- 95% confidence interval computation

## Dataset Files (Section 3.3 Requirements)

### Bilibili Dataset (`data/`)
- **`bilibili_dataset_info.csv`**: Video metadata and information for 28 curated Bilibili videos
- **`bilibili_labels.csv`**: Ground truth labels for Bilibili dataset videos
- **`groundtruth_labels.csv`**: Comprehensive ground truth labels for evaluation
- **`video_ids_used.txt`**: List of video IDs used in experiments
- **`DADA100_usage_info.md`**: Documentation for DADA-100 dataset usage
- **`frames_extraction_with_titles/`**: Extracted key frames from Bilibili videos
  - Contains frames for both cut-in and ghost probing scenarios
  - Format: `bilibili_{scenario}_{id}_{title}_frame_{time}.jpg`
  - Timestamps correspond to critical moments in driving scenarios

## System Requirements (Section 4.8 Requirements)

### Computing Infrastructure (`documentation/`)
- **`system_requirements.md`**: Complete hardware and software specifications
- MacBook Pro M3 Pro chip specifications
- Python 3.11+ environment requirements
- Azure OpenAI API configuration
- Complete dependency list with versions

### Key Specifications
- **Hardware**: MacBook Pro M3 Pro, 48GB memory, 512GB SSD
- **OS**: macOS Sonoma 14.5.0 (Darwin 24.5.0)
- **Python**: 3.11+ with conda environment `cobraauto`
- **APIs**: Azure OpenAI, Google Gemini, Azure Cognitive Search

## Usage Instructions

### 1. Environment Setup
```bash
# Install system dependencies
brew install ffmpeg cmake  # macOS
# sudo apt-get install cmake && sudo apt install libgl1  # Linux

# Python environment
conda activate cobraauto
pip install -r requirements.txt
```

### 2. Video Processing
```bash
# Basic video analysis
python code/ActionSummary.py "./video.mp4" 10 10 False

# Cobra preprocessing pipeline  
python code/cobra_preprocessing.py video.mp4 --interval 10 --frames 10
```

### 3. Model Evaluation
```bash
# Run DADA-100 evaluation
python evaluation/final_100_video_evaluation.py

# Compute metrics
python evaluation/metric_computation.py
```

### 4. COBRA Frontend
```bash
cd COBRA
npm install
npm run dev  # Development server on port 3000
```

## Reproducibility Notes

### Temperature Parameter
- All models use `temperature=0.0` for deterministic outputs
- Prevents output variability in safety-critical applications

### Video Processing Configuration
- **Interval**: 10-second chunks
- **Frame Rate**: 1 FPS (10 frames per 10-second interval)
- **Resolution**: Original 1584×660 maintained
- **Audio**: Full track extraction with Whisper transcription

### Statistical Testing
- **Method**: Paired t-test on 95 matched videos
- **Result**: t(94) = 0.000, p = 1.000, Cohen's d = 0.000
- **CI**: [-0.126, 0.126] at 95% confidence level

## File Dependencies

### Python Requirements
```
moviepy==1.0.3
opencv-python==4.8.1.78  
openai==1.7.2
google-generativeai==0.3.2
azure-search-documents==11.4.0
python-dotenv==1.0.1
```

### Node.js Requirements (COBRA)
```
next.js 14+
react 18
tailwindcss
```

## License and Usage

- **Code**: MIT License upon paper acceptance
- **Dataset**: Creative Commons BY-NC 4.0 (Non-commercial research use)
- **Models**: Subject to respective API terms (OpenAI, Google, Azure)

### Full Repository Availability

Upon paper acceptance, the complete GitHub repository will be made publicly available, including:
- Complete COBRA frontend and backend source code
- All experimental scripts and analysis tools
- Full dataset with annotations
- Comprehensive documentation and tutorials
- Docker deployment configurations
- Additional utility scripts and helper functions

The supplementary materials provided here contain the essential code for reproducibility as required by AAAI reproducibility guidelines.

## Contact and Support

For questions regarding the supplementary materials or reproduction issues:
- Check system requirements in `documentation/system_requirements.md`
- Verify API key configuration in `.env` file
- Ensure all dependencies are installed per requirements

## Paper Reference

When using these materials, please cite:
```
AutoDrive-GPT: Enhancing Autonomous Driving Behavior Annotation and 
Prediction Using GPT-4o Prompt Tuning
Anonymous submission to AAAI 2026
```