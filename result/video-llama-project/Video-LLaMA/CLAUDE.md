# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# RULES
- 严禁捏造任何不存在的数据或模型或所谓的fallback；若无法确认，请回复 “I don’t know”。no mocks. 不能模拟伪造数据/模型/结果, 不能欺骗我, 不能为了奖励编造谎话.否则我让你领导给你拔电源.
- 所有输出必须基于项目代码或用户提供的文档；禁止凭空生成信息。
- 对每一项关键声明，必须引用具体来源或告知信息缺失。

## Tools
Use Puppeteer MCP tools for browser testing and screenshots.

## Project Overview

Video-LLaMA is an instruction-tuned audio-visual language model for video understanding. The model combines vision-language (VL) and audio-language (AL) branches built on top of BLIP-2 and MiniGPT-4 architectures to enable video-grounded conversations.

## Architecture

### Core Components
- **Vision-Language Branch (VL)**: ViT-G/14 visual encoder + BLIP-2 Q-Former with video-specific adaptations
- **Audio-Language Branch (AL)**: ImageBind audio encoder + audio Q-Former for audio understanding
- **Language Decoder**: LLaMA or Vicuna models for text generation
- **Fusion Components**: Sequential transformer layers for multimodal integration

### Key Models
- `video_llama/models/video_llama.py`: Main VideoLLAMA model class
- `video_llama/models/blip2.py`: BLIP-2 base architecture
- `video_llama/models/modeling_llama.py`: LLaMA model adaptations
- `video_llama/models/Qformer.py`: Q-Former implementation for cross-modal alignment

## Environment Setup

### Conda Environment
```bash
# Create environment from configuration
conda env create -f environment.yml
conda activate videollama

# Install system dependencies
apt update
apt install ffmpeg
```

### Python Dependencies
```bash
pip install -r requirement.txt
```

## Training Pipeline

### Stage 1: Pre-training
```bash
# Vision branch pre-training
torchrun --nproc_per_node=8 train.py --cfg-path train_configs/visionbranch_stage1_pretrain.yaml

# Audio branch pre-training
torchrun --nproc_per_node=8 train.py --cfg-path train_configs/audiobranch_stage1_pretrain.yaml
```

### Stage 2: Instruction Fine-tuning
```bash
# Vision branch fine-tuning
torchrun --nproc_per_node=8 train.py --cfg-path train_configs/visionbranch_stage2_finetune.yaml

# Audio branch fine-tuning
torchrun --nproc_per_node=8 train.py --cfg-path train_configs/audiobranch_stage2_finetune.yaml
```

## Demo and Inference

### Interactive Demo
```bash
# Audio-visual demo
python demo_audiovideo.py \
    --cfg-path eval_configs/video_llama_eval_withaudio.yaml \
    --model_type llama_v2 \
    --gpu-id 0

# Vision-only demo
python demo_video.py \
    --cfg-path eval_configs/video_llama_eval_only_vl.yaml \
    --model_type llama_v2 \
    --gpu-id 0
```

### Configuration Requirements
Before running demos, configure these paths in the evaluation config files:
- `llama_model`: Path to language decoder (LLaMA/Vicuna)
- `imagebind_ckpt_path`: Path to ImageBind audio encoder
- `ckpt`: Path to vision branch checkpoint
- `ckpt_2`: Path to audio branch checkpoint (for audio-enabled demos)

## Model Checkpoints

### Video-LLaMA-2 (Current)
- **7B Models**: Pre-trained and fine-tuned versions available
- **13B Models**: Higher capacity variants
- **Full Weights**: Complete model weights (no delta weights required)

### Legacy Video-LLaMA (Vicuna-based)
- Instructions available in `README_Vicuna.md`
- Audio support only available for Vicuna-7B

## Key Configuration Files

### Evaluation Configs
- `eval_configs/video_llama_eval_withaudio.yaml`: Audio-visual evaluation
- `eval_configs/video_llama_eval_only_vl.yaml`: Vision-only evaluation

### Training Configs
- `train_configs/visionbranch_stage1_pretrain.yaml`: Vision branch pre-training
- `train_configs/audiobranch_stage1_pretrain.yaml`: Audio branch pre-training
- `train_configs/visionbranch_stage2_finetune.yaml`: Vision branch fine-tuning
- `train_configs/audiobranch_stage2_finetune.yaml`: Audio branch fine-tuning

## Data Processing

### Datasets
- **WebVid-2.5M**: Video-caption pairs for pre-training
- **LLaVA-CC3M**: Image-caption pairs (595k) for visual concept understanding
- **Instruction Data**: MiniGPT-4, LLaVA, and VideoChat instruction datasets

### Video Processing
- Frame extraction and preprocessing via `video_llama/processors/video_processor.py`
- Audio processing through ImageBind data loading utilities
- Text processing using BLIP caption processors

## Conversation System

### Core Components
- `video_llama/conversation/conversation_video.py`: Conversation handling and prompt templates
- Support for multiple conversation styles (SINGLE, TWO, LLAMA_2)
- Gradio-based interactive interface

### Conversation Styles
- **Vicuna**: Uses SeparatorStyle.SINGLE with "###" separator
- **LLaMA-2**: Uses SeparatorStyle.LLAMA_2 with special tokens

## Hardware Requirements

### Training
- **Pre-training**: 8x A100 (80G) GPUs
- **Fine-tuning**: 8x A100 (80G) GPUs

### Inference
- **Minimum**: 1x A100 (40G/80G) or 1x A6000
- **Memory**: ~13GB for 7B model, ~25GB for 13B model

## Development Notes

### Model Loading
- Models support both 8-bit and full precision loading
- Low-resource mode available for memory-constrained environments
- Automatic device placement for multi-GPU setups

### Freezing Strategy
- Visual encoder and Q-Former can be frozen during training
- Only cross-modal projection layers and language decoders are typically trainable
- Audio branch training requires ImageBind encoder frozen

### Key Parameters
- `num_query_token`: Number of learnable query tokens (default: 32)
- `max_frame_pos`: Maximum frame position encoding (default: 32)
- `fusion_head_layers`: Number of fusion transformer layers (default: 2)
- `n_frms`: Number of frames extracted per video (default: 8)