# System Requirements and Computing Infrastructure

## Hardware Specifications
- **System**: MacBook Pro with Apple M3 Pro chip
- **CPU**: 10-core CPU
- **GPU**: 16-core GPU  
- **Memory**: 48GB unified memory
- **Storage**: 512GB SSD
- **Operating System**: macOS Sonoma 14.5.0 (Darwin 24.5.0)

## Software Dependencies

### Python Environment
```
Python 3.11+
conda activate cobraauto
```

### Core Dependencies (requirements.txt)
```
moviepy==1.0.3
opencv-python==4.8.1.78
openai==1.7.2
google-generativeai==0.3.2
azure-search-documents==11.4.0
python-dotenv==1.0.1
azure-cognitiveservices-speech==1.34.0
requests==2.31.0
tqdm==4.66.1
jinja2==3.1.2
```

### System Prerequisites
```bash
# macOS
brew install ffmpeg cmake

# Linux
sudo apt-get install cmake
sudo apt install libgl1
```

### Frontend Dependencies (COBRA)
```bash
cd COBRA
npm install
```

## Cloud Services Configuration

### Azure OpenAI
- **API Types**: AUDIO_API_TYPE, VISION_API_TYPE ("Azure" or "OpenAI")
- **Keys**: AZURE_SPEECH_KEY, AZURE_WHISPER_KEY, AZURE_VISION_KEY
- **Endpoints**: AZURE_WHISPER_ENDPOINT, VISION_ENDPOINT, VISION_DEPLOYMENT_NAME

### OpenAI
- **API Key**: OPENAI_API_KEY

### Google Gemini
- **API Key**: GEMINI_API_KEY
- **Model**: GEMINI_MODEL ("gemini-1.5-flash")

### Azure Cognitive Search
- **Endpoints**: SEARCH_ENDPOINT, SEARCH_API_KEY, INDEX_NAME

## Performance Characteristics
- **Temperature Parameter**: 0.0 (deterministic outputs)
- **Video Processing**: 10-second intervals, 10 frames per interval (1 FPS)
- **Resolution**: Maintained original 1584×660 resolution
- **API Processing**: Cloud-based AI through Azure OpenAI APIs
- **Local Storage**: Sufficient space for video processing temporary files

## Memory and Processing Requirements
- **Minimum RAM**: 16GB (48GB recommended for large batch processing)
- **Temporary Storage**: ~2GB per video during processing (auto-cleaned)
- **Network**: Stable internet connection for cloud API calls
- **Processing Time**: ~30-60 seconds per 10-second video interval depending on API response times