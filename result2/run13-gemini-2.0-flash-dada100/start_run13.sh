#!/bin/bash

# Run 13: Gemini 2.0 Flash DADA-100 Analysis
# 启动脚本

echo "=========================================="
echo "Run 13: Gemini 2.0 Flash DADA-100 Analysis"
echo "=========================================="

# 检查Python环境
python3 --version

# 检查环境变量
if [[ -z "$GEMINI_API_KEY" ]]; then
    echo "⚠️  GEMINI_API_KEY not found, loading from .env file"
    export $(cat /Users/wanmeng/repository/GPT4Video-cobra-auto/.env | grep -v ^# | xargs)
fi

echo "🔑 Gemini API Key: ${GEMINI_API_KEY:0:10}..."
echo "🤖 Gemini Model: $GEMINI_MODEL"

# 切换到脚本目录
cd "$(dirname "$0")"

# 创建logs目录
mkdir -p logs

echo "📁 Output Directory: $(pwd)"
echo "🎬 Video Source: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/"

# 启动分析
echo ""
echo "🚀 Starting Gemini 2.0 Flash Analysis..."
echo "⏰ $(date)"
echo ""

python3 run13_gemini_2_0_flash_dada100.py

echo ""
echo "🏁 Analysis completed at $(date)"
echo "📊 Check the summary JSON file for results"
echo "📝 Check logs/ directory for detailed logs"