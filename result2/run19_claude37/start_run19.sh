#!/bin/bash

# Run 19: Claude 4 Ghost Probing Detection with Few-shot Learning (100 Videos)
# 启动脚本

echo "🚀 启动 Run 19: Claude 4 Ghost Probing Detection 实验"

# 确保conda环境激活
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cobraauto

# 设置工作目录
cd /Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run19_claude4

# 运行实验
python run19_claude4_ghost_probing_fewshot_100videos.py

echo "✅ Run 19 实验完成！"