#!/bin/bash

echo "🚀 启动Run8-Rerun1和Run8-Rerun2实验..."

# 启动Run8-Rerun1
echo "📍 启动Run8-Rerun1实验 (100个DADA视频)"
cd /Users/wanmeng/repository/GPT4Video-cobra-auto
nohup python result2/run8-rerun1/run8_rerun_plus_image_fewshot.py > result2/run8-rerun1/experiment_output.log 2>&1 &
RUN1_PID=$!
echo "Run8-Rerun1 PID: $RUN1_PID"

# 等待5分钟避免API限制冲突
echo "⏳ 等待5分钟避免API冲突..."
sleep 300

# 启动Run8-Rerun2
echo "📍 启动Run8-Rerun2实验 (100个DADA视频)"
nohup python result2/run8-rerun2/run8_rerun_plus_image_fewshot.py > result2/run8-rerun2/experiment_output.log 2>&1 &
RUN2_PID=$!
echo "Run8-Rerun2 PID: $RUN2_PID"

echo "🎯 两个实验已启动!"
echo "Run8-Rerun1 PID: $RUN1_PID (日志: result2/run8-rerun1/experiment_output.log)"
echo "Run8-Rerun2 PID: $RUN2_PID (日志: result2/run8-rerun2/experiment_output.log)"

echo "📊 使用以下命令监控进度:"
echo "tail -f result2/run8-rerun1/experiment_output.log"
echo "tail -f result2/run8-rerun2/experiment_output.log"

echo "🔍 检查进程状态:"
echo "ps aux | grep run8_rerun_plus_image_fewshot"