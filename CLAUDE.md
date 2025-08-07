# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# NO MOCK RULES
- Strictly no fabrication of non-existent data, models, or so-called “fallbacks”; no “mock data”, no placeholder, “proof of concept,” and “simulation” are forbidden. If you cannot verify something, reply “I don’t know.” No mocks. Do not simulate or invent data/models/results, do not deceive me, and do not lie for rewards—otherwise I’ll have your boss pull the plug.
- All output must be based on the project’s code or user-provided documentation; inventing information out of thin air is prohibited.
- Cite the exact file path for every code change.

## Tools
Use Puppeteer MCP tools for browser testing and screenshots.

## Prerequisites

### System Requirements
- Python 3.11+ for backend (currently using Python 3.11 based on requirements.txt)
- Node.js 18+ for frontend
- CMake (required before pip install)
  - macOS: `brew install ffmpeg cmake`
  - Linux: `sudo apt-get install cmake` and `sudo apt install libgl1` (for OpenGL)

### Environment Setup
- **Conda Environment**: `conda activate cobraauto` (required for all Python scripts and experiments)

### Important Notes
- Use GPT-4o instead of GPT-4-turbo vision for latest video interpretation capability
- Azure GPT4 Vision service limitations:
  - Max 20 images per call (unstable)
  - Max FPI (frames per interval) is 10
  - May need to apply to turn off content filtering (adds 30+ seconds to each call)

## Project Overview

GPT4Video-cobra-auto is a comprehensive video analysis system focusing on autonomous driving video analysis, particularly "ghost probing" detection. The project combines AI-powered video processing with a web-based interface, using computer vision, natural language processing, and large language models to analyze video content from the DADA-2000 dataset for research and safety applications.

## COBRA Frontend Application

COBRA (Cobra-Content Based Video Retrieval Analysis) is the web-based frontend interface that provides researchers and users with an intuitive way to interact with the video analysis system. Built with Next.js, COBRA serves as the primary user interface for exploring, analyzing, and visualizing driving video analysis results.

### Key Features

**Interactive Video Player**
- Custom-built video player component with analysis results overlay
- Real-time display of AI-generated annotations and predictions
- Timeline-based navigation with frame-level precision
- Support for multiple video formats (.avi, .mp4, .mov)
- Synchronized display of analysis results with video playback

**Semantic Search Interface**
- Advanced search functionality powered by Azure Cognitive Search
- Vector-based semantic queries for finding specific driving scenarios
- Filters by video categories, timestamps, and analysis results
- Integration with the DADA-2000 dataset indexing system
- Real-time search with relevance scoring and result ranking

**Analysis Results Visualization**
- Comparative display of results from multiple AI models (GPT-4, Gemini, DriveMM, LlaVA, WiseAD)
- Interactive charts and graphs for model performance metrics
- Side-by-side comparison views for research validation
- Export functionality for analysis data and visualizations

**Research Dashboard**
- Batch processing status monitoring and progress tracking
- Statistical analysis and model evaluation tools
- Ghost probing detection results visualization
- Academic research workflow integration for AAAI26 paper development

### COBRA Workflow Integration

1. **Data Ingestion**: Automatically indexes processed video analysis results from the backend
2. **Search & Discovery**: Enables researchers to find specific driving scenarios using natural language queries
3. **Analysis Review**: Provides detailed views of AI model predictions and comparisons
4. **Research Export**: Facilitates data export for academic research and paper preparation
5. **Model Validation**: Supports ground truth comparison and statistical analysis workflows

### Technical Architecture

**Frontend Stack**
- **Framework**: Next.js 14+ with React 18
- **Styling**: Tailwind CSS for responsive design
- **Components**: Custom video player, search interface, data visualization charts
- **API Integration**: RESTful APIs connecting to Python backend services

**Key API Endpoints**
- `/api/chat/`: Natural language query processing and response generation
- `/api/cog/`: Azure Cognitive Search integration for semantic video search
- Real-time WebSocket connections for batch processing updates

**Deployment**
- Docker containerization for consistent deployment
- Azure VM hosting with port 3000 configuration
- Environment-specific configuration management

### Usage in Research Context

COBRA is specifically designed to support academic research in autonomous driving safety:
- **Comparative Model Analysis**: Visual comparison of different AI model predictions
- **Ghost Probing Research**: Specialized interface for dangerous driving maneuver detection
- **DADA-2000 Dataset Exploration**: Comprehensive navigation and analysis of the 2000+ video dataset
- **Statistical Validation**: Built-in tools for F1-score calculation, precision/recall analysis
- **Academic Publication Support**: Data export and visualization tools for research papers

## Architecture

### Core System Design
The system processes video datasets (primarily DADA-2000 format) through multiple AI models for comparative analysis:

**Backend (Python)**
- **Video Processing Pipeline**: Frame extraction from .avi files, scene analysis at configurable intervals
- **Multi-Model AI Integration**: GPT-4 Vision, Google Gemini, Azure services, and specialized models (DriveMM, LlaVA, WiseAD)
- **Audio Processing**: Speech-to-text transcription using Azure Whisper or OpenAI
- **Batch Processing**: Handles large-scale video analysis with checkpoint/resume capabilities
- **Results Management**: Organized output structure in `result/` directory by model type

**Frontend (Next.js - COBRA)**
- **Video Player Component**: Custom player with analysis results overlay located in `COBRA/components/`
- **Search Interface**: Semantic search functionality via Azure Cognitive Search
- **API Routes**: `/api/chat/` and `/api/cog/` for backend integration at `COBRA/app/api/`
- **Data Visualization**: Analysis results display and comparison tools

### Key Directory Structure
- **Root Scripts**: Core video analysis modules (`ActionSummary.py`, `ChapterAnalyzer-zh.py`, etc.)
- **COBRA/**: Next.js frontend application with video player and search interface
- **DADA-2000-videos/**: Main video dataset (2000+ driving videos in `images_XX_YYY.avi` format)
- **result/**: Analysis results organized by model (`gpt-4o/`, `gemini-1.5-flash/`, `DriveMM/`, etc.)
- **fsl/**: Few-shot learning templates and examples
- **Documentation/**: Research reports and analysis summaries
- **AAAI26_paper/**: Academic paper source files

## Key Scripts and Their Purpose

### Core Video Analysis Scripts (Root Directory)
- `ActionSummary.py`: General video analysis with configurable interval/frame parameters
- `ChapterAnalyzer-zh.py`: Chapter-based video analysis with Chinese language support
- `SceneAnalyzer.py`: Scene-level video analysis and breakdown
- `ingest_scenes.py`: Ingests processed video data into Azure Cognitive Search for semantic search
- `video_utilities.py`: Common video processing utilities and helper functions

### Model-Specific Analysis Scripts (result/ subdirectories)
- `result/gemini-1.5-flash/ActionSummary-gemini.py`: Google Gemini model analysis
- `result/gpt-4o/`: GPT-4 Vision analysis scripts and results
- `result/DriveMM/`: DriveMM model analysis and evaluation
- `result/LlaVA/`: LlaVA model implementation and analysis
- `result/WiseAD/`: WiseAD model integration and evaluation
- `result/comparison/`: Cross-model comparison and evaluation scripts

### Research and Analysis Tools
- `result/comparison/final_100_video_evaluation.py`: Comprehensive model comparison
- `result/artifacts/`: Utility scripts for dataset creation and deployment checks
- `Documentation/`: Contains final analysis reports and research summaries
- From now on, put the python file names used when adding a run into model_run_log.md

## Common Development Commands

### Backend Setup and Analysis
```bash
# Install dependencies and setup
pip install -r requirements.txt
cp .envsample .env
rm -rf frames/ audio/  # Clean temporary dirs before processing

# Core video analysis scripts
python ActionSummary.py "./video.mp4" 10 10 False
python ChapterAnalyzer-zh.py
python ingest_scenes.py

# Model-specific analysis
python result/gemini-1.5-flash/ActionSummary-gemini.py
python result/comparison/final_100_video_evaluation.py

# Batch processing with advanced options
python ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper-batch.py --folder DADA-2000-videos --limit 5
python ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper-batch.py --start-at 5  # resume from video 5
python ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper-batch.py --retry-failed
```

### Frontend Development (COBRA)
```bash
cd COBRA
npm install
npm run dev      # Development server (port 3000)
npm run build    # Production build
npm run lint     # Code linting
```

### Docker Deployment
```bash
docker-compose up --build
```

## Environment Configuration

Copy `.envsample` to `.env` and configure:

### Backend (.env)
- **API Types**: `AUDIO_API_TYPE`, `VISION_API_TYPE` ("Azure" or "OpenAI")
- **Azure Keys**: `AZURE_SPEECH_KEY`, `AZURE_WHISPER_KEY`, `AZURE_VISION_KEY`
- **Azure Endpoints**: `AZURE_WHISPER_ENDPOINT`, `VISION_ENDPOINT`, `VISION_DEPLOYMENT_NAME`
- **OpenAI**: `OPENAI_API_KEY`
- **Gemini**: `GEMINI_API_KEY`, `GEMINI_MODEL` ("gemini-1.5-flash")

### Frontend (COBRA/.env)
- **Azure OpenAI**: `AZ_OPENAI_KEY`, `AZ_OPENAI_BASE`, `AZ_OPENAI_REG`, `AZ_OPENAI_VERSION`
- **Search**: `SEARCH_ENDPOINT`, `SEARCH_API_KEY`, `INDEX_NAME`
- **Model**: `GPT4` (e.g., "4turbo")

## Video Processing Workflow

1. **Frame Extraction**: Videos are processed to extract frames at specified intervals
2. **AI Analysis**: Frames are analyzed using GPT-4 Vision or Gemini for:
   - Action detection and prediction
   - Object and character identification
   - Scene understanding
   - Safety analysis (for driving videos)
3. **Audio Processing**: Audio tracks are transcribed using Whisper
4. **Data Integration**: Results are combined and stored in JSON format
5. **Search Integration**: Processed data is ingested into Azure Cognitive Search with vector embeddings

## Key Data Structures

### ActionSummary JSON Format
```json
{
  "timestamp": "time_range",
  "summary": "scene_description",
  "actions": "current_actions",
  "characters": "people_present",
  "key_objects": "important_objects",
  "key_actions": "significant_actions",
  "next_action": "predicted_next_action"
}
```

## Key Data Directories

### Video Processing
- `DADA-2000-videos/`: Main dataset directory with 2000+ videos named `images_XX_YYY.avi`
- `result/DADA-100-videos/`: Subset of 100 videos for focused analysis with ground truth labels

### Results and Outputs
- `result/`: Organized by model type:
  - `gpt-4o/`: GPT-4 Vision analysis results and scripts
  - `gemini-1.5-flash/`: Google Gemini analysis results (1000+ JSON files)
  - `DriveMM/`: DriveMM model results and Azure ML configurations
  - `LlaVA/`: LlaVA model analysis and evaluation scripts
  - `WiseAD/`: WiseAD model integration results
  - `comparison/`: Cross-model comparison analyses and visualizations
  - `artifacts/`: Deployment scripts and utilities
- `logs/`: Detailed processing logs and raw JSON responses
- `Documentation/`: Research reports, final summaries, and AAAI26 paper materials

### Working Directories
- `fsl/`: Few-shot learning examples and templates (jinja2 templates, example frames)
- `labelresult/`: Ground truth labels for evaluation (cutin/ghosting annotations)

### Temporary Directories
- `frames/`: Temporary extracted video frames (auto-cleaned after processing)
- `audio/`: Temporary extracted audio files (auto-cleaned after processing)

## Batch Processing Architecture

The system supports advanced batch processing with:
- **Checkpoint/Resume**: `--start-at N` to resume from video N
- **Failure Recovery**: `--retry-failed` to retry failed videos  
- **Progress Tracking**: Real-time progress with tqdm, saves stats to JSON
- **Video ID Extraction**: Supports DADA-2000 (`images_XX_YYY.avi`), bilibili, and generic formats
- **Logging**: Multi-level logging with `--log-level DEBUG/INFO/WARNING/ERROR`

### Key Command Line Arguments
```bash
--folder DADA-2000-videos  # Video directory
--interval 10              # Seconds per interval
--frames 10               # Frames per interval  
--limit 5                 # Process only N videos
--start-at 10             # Resume from video 10
--retry-failed            # Retry previously failed videos
--no-skip                 # Don't skip processed videos
```

## Development Notes

### Frame Processing
- Frame intervals (FI) and frames per interval (FPI) can be adjusted for different analysis depths
- Higher FPI provides more detailed analysis but increases processing time
- **Standard FPS**: Current experiments use FPS (Frames Per Second) = 1, extracting 10 frames per 10-second interval
- Temporary `frames/` and `audio/` directories are created during processing
- Complete video processing: calculates intervals based on total video duration
- Auto-adjusts frames for partial final segments

### API Integration
- The system supports both Azure and OpenAI APIs
- Gemini integration is available for additional AI capabilities
- Rate limiting and retry mechanisms are implemented for API calls
- JSON parsing with error handling and response validation
- **Temperature Parameter**: All AI model calls now use temperature=0 for consistent, deterministic results (updated 2025-07-26)

### Search Functionality
- Vector embeddings are generated for semantic search
- Search results include relevance scoring
- Multiple search configurations are supported

### Model Evaluation
- Comparative analysis between GPT-4o and Gemini models
- Ground truth labels loaded from `result/labels.csv`
- Time-based matching with configurable tolerance
- Statistical analysis with visualization outputs

## Testing Video Processing

### Single Video Testing
1. Ensure environment variables are set correctly
2. Place test videos in the root directory or specify full paths
3. Run the appropriate analysis script based on your use case
4. Check the generated JSON files for results
5. Verify that `frames/` and `audio/` directories are cleaned up after processing

### Batch Processing Testing
1. Use the DADA-2000-videos dataset for comprehensive testing
2. Start with a small subset using `--limit 5` parameter
3. Monitor progress in `logs/` directory for detailed processing information
4. Check `result/` directory for organized output files
5. Use evaluation scripts to compare model performance

### Video Format Support
- Primary format: `.avi` files with DADA-2000 naming convention
- Supported: `.mp4`, `.mov` files
- Expected naming: `images_XX_YYY.avi` where XX is category, YYY is sequence number

## Troubleshooting

### Common Issues
- **Dependencies**: `pip install -r requirements.txt` and ensure CMake installed (`brew install cmake` on macOS)
- **Memory issues**: Reduce FPI (frames per interval) for large videos
- **API rate limits**: Adjust retry mechanisms in scripts
- **Frontend errors**: Delete `.next` folder and rebuild; ensure port 3000 available
- **Azure VM**: Add inbound policy (port 3000, TCP, Allow, priority=100)
- **Vector search**: Match semantic configuration name in `COBRA/app/api/cog/route.js:49` to index creation

### Performance Tips
- Use Gemini for faster batch processing
- Adjust FI/FPI ratios based on analysis depth needed
- DADA-2000 stats: 60% videos >10s, average ~11.46s duration

## Project Use Cases

### Primary Research Applications
- **Ghost Probing Detection**: Identifying dangerous "ghost probing" maneuvers in autonomous driving scenarios
- **Multi-Model Comparison**: Comparative analysis of GPT-4, Gemini, DriveMM, LlaVA, and WiseAD models
- **Autonomous Driving Safety**: Video analysis for safety-critical driving scenario detection
- **Academic Research**: AAAI26 paper development and DADA-2000 dataset analysis
- **Model Evaluation**: Performance metrics calculation and statistical analysis across models

### Technical Applications
- **Batch Video Processing**: Large-scale analysis of driving video datasets
- **Semantic Video Search**: Azure Cognitive Search integration for research queries
- **Few-Shot Learning**: Template-based analysis using driving scenario examples
- **Cross-Model Validation**: Comparing model predictions for research reliability

## Video Processing Workflow for Research

### DADA-2000 Analysis Pipeline
1. **Video Dataset**: Process videos from `DADA-2000-videos/` with naming format `images_XX_YYY.avi`
2. **Frame Extraction**: Extract frames at specified intervals (typically 10s intervals, 10 frames per interval)
3. **Multi-Model Analysis**: Run the same video through multiple AI models (GPT-4, Gemini, DriveMM, etc.)
4. **Result Storage**: Store JSON results in model-specific directories under `result/`
5. **Comparative Analysis**: Use scripts in `result/comparison/` to evaluate model performance
6. **Ground Truth Comparison**: Compare against labels in `result/DADA-100-videos/groundtruth_labels.csv`

### Research Focus Areas
- **Ghost Probing Detection**: Focus on dangerous cut-in maneuvers from blind spots
- **Model Performance**: F1-scores, precision, recall analysis across models
- **Statistical Validation**: 95% confidence intervals and significance testing
- **Academic Publication**: Results feeding into AAAI26 conference paper

## Key Dependencies
- **moviepy==1.0.3**: Video/frame processing
- **opencv-python==4.8.1.78**: Computer vision  
- **openai==1.7.2**: GPT-4 Vision API
- **google-generativeai==0.3.2**: Gemini API
- **azure-search-documents==11.4.0**: Semantic search
- **python-dotenv==1.0.1**: Environment management

# Summary instructions
When compacting, please prioritize recent code diffs and test outputs over chat history.
# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.