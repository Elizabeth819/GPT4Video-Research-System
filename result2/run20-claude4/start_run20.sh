#!/bin/bash

# Run 20: Claude 4 Ghost Probing Detection with Few-shot Learning (100 Videos)
# 启动脚本

echo "🚀 启动 Run 20: Claude 4 (claude-sonnet-4-20250514) Ghost Probing Detection 实验"

# 确保conda环境激活
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cobraauto

# 设置工作目录
cd /Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run20-claude4

# 运行实验
python run20_claude4_ghost_probing_fewshot_100videos.py

echo "✅ Run 20 实验完成！"