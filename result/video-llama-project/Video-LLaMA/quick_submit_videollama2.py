#!/usr/bin/env python3
"""
Quick Submit Video-LLaMA2 Ghost Probing Job
快速提交Video-LLaMA2鬼探头检测作业到Azure ML
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置Azure ML环境变量
os.environ["AZURE_SUBSCRIPTION_ID"] = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a"
os.environ["AZURE_RESOURCE_GROUP"] = "video-llama2-ghost-probing-rg"
os.environ["AZURE_WORKSPACE_NAME"] = "video-llama2-ghost-probing-ws"

def main():
    """主函数"""
    print("🚀 Quick Submit Video-LLaMA2 Ghost Probing Job")
    print("=" * 60)
    print(f"Azure 订阅: {os.environ['AZURE_SUBSCRIPTION_ID']}")
    print(f"资源组: {os.environ['AZURE_RESOURCE_GROUP']}")
    print(f"工作区: {os.environ['AZURE_WORKSPACE_NAME']}")
    print("=" * 60)
    
    # 显示将要处理的内容
    print("📋 任务概述:")
    print("   🎬 模型: Video-LLaMA2")
    print("   📹 视频: DADA-2000数据集 (images_1_001 到 images_5_XXX)")
    print("   🎯 目标: 100个视频的鬼探头检测")
    print("   💻 平台: Azure ML A100 GPU")
    print("   📊 输出: 与GPT-4.1格式一致的JSON结果")
    print("=" * 60)
    
    # 询问用户确认
    choice = input("是否继续提交作业? (y/n): ").strip().lower()
    
    if choice != 'y':
        print("❌ 作业提交已取消")
        return
    
    try:
        # 首先进行环境检查
        print("\n🔍 Step 1: 环境检查...")
        result = subprocess.run([
            sys.executable, "submit_videollama2_ghost_probing_job.py", "--check-only"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ 环境检查失败")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return
        
        print("✅ 环境检查通过")
        
        # 提交作业
        print("\n🚀 Step 2: 提交Azure ML作业...")
        result = subprocess.run([
            sys.executable, "submit_videollama2_ghost_probing_job.py", "--no-monitor"
        ], capture_output=True, text=True)
        
        print("作业提交结果:")
        print(result.stdout)
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ 作业提交成功!")
            print("\n📋 后续步骤:")
            print("1. 在Azure ML Studio中监控作业进度")
            print("2. 作业完成后下载结果")
            print("3. 使用以下命令监控作业:")
            print("   python submit_videollama2_ghost_probing_job.py --monitor-only <job_name>")
            print("4. 使用以下命令下载结果:")
            print("   python submit_videollama2_ghost_probing_job.py --download-only <job_name>")
        else:
            print("❌ 作业提交失败")
            
    except Exception as e:
        logger.error(f"❌ 执行失败: {e}")

if __name__ == "__main__":
    main()