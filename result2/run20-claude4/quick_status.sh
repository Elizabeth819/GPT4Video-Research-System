#!/bin/bash

# Quick status check for Run 20 Claude 4 experiment
echo "🎯 Run 20 - Claude 4 Ghost Probing Detection Quick Status"
echo "⏰ 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if process is running
if pgrep -f "run20_claude4_ghost_probing_fewshot_100videos.py" > /dev/null; then
    echo "✅ 实验状态: 正在运行"
    PROCESS_ID=$(pgrep -f "run20_claude4_ghost_probing_fewshot_100videos.py")
    echo "🔧 进程ID: $PROCESS_ID"
else
    echo "❌ 实验状态: 已停止"
fi

echo ""

# Check for latest intermediate results
cd /Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run20-claude4

if ls run20_intermediate_*videos_*.json 1> /dev/null 2>&1; then
    LATEST_FILE=$(ls -t run20_intermediate_*videos_*.json | head -n1)
    echo "📁 最新结果: $LATEST_FILE"
    
    # Extract video count from filename
    VIDEO_COUNT=$(echo $LATEST_FILE | sed -n 's/.*_\([0-9]\+\)videos_.*/\1/p')
    echo "📈 已处理: $VIDEO_COUNT/100 视频 ($(echo "scale=1; $VIDEO_COUNT * 100 / 100" | bc)%)"
else
    echo "📊 中间结果: 暂无（可能刚开始运行）"
fi

echo ""

# Check log file for latest activity
if ls run20_claude4_ghost_probing_fewshot_*.log 1> /dev/null 2>&1; then
    LATEST_LOG=$(ls -t run20_claude4_ghost_probing_fewshot_*.log | head -n1)
    echo "📝 最新日志活动:"
    tail -3 $LATEST_LOG | grep -E "(处理视频|检测=|评估=)" | tail -1
fi

echo ""
echo "🔍 详细监控: python monitor_run20.py"