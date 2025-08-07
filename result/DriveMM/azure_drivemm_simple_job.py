#!/usr/bin/env python3
"""
简化的DriveMM Azure A100作业提交脚本
"""

import os
import sys
import subprocess
from datetime import datetime

def create_simple_job():
    """创建简化的DriveMM作业配置"""
    
    job_yaml = f"""$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
code: ./
command: python azure_real_drivemm_simple.py
environment: azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10
compute: drivemm-a100-cluster
experiment_name: drivemm_simple_analysis
display_name: DriveMM_Simple_DADA2000_{datetime.now().strftime('%Y%m%d_%H%M%S')}
description: Simplified DriveMM analysis on DADA-2000 videos using Azure A100 GPU
tags:
  model: DriveMM-Simplified
  dataset: DADA-2000
  task: ghost_probing_detection
  gpu: A100-80GB
  mode: simplified_analysis
"""
    
    with open("drivemm_simple_job.yml", "w") as f:
        f.write(job_yaml)
    
    return "drivemm_simple_job.yml"

def submit_simple_job():
    """提交简化的DriveMM作业"""
    print("🚀 提交简化的DriveMM Azure A100作业...")
    
    try:
        # 创建作业配置
        job_file = create_simple_job()
        print(f"✅ 作业配置文件: {job_file}")
        
        # 检查简化脚本是否存在
        if not os.path.exists("azure_real_drivemm_simple.py"):
            print("❌ 简化脚本不存在")
            return None
        
        print(f"✅ 处理器脚本: azure_real_drivemm_simple.py")
        
        # 提交作业
        cmd = [
            "az", "ml", "job", "create",
            "--file", job_file,
            "--workspace-name", "drivelm-ml-workspace",
            "--resource-group", "drivelm-rg"
        ]
        
        print("🔄 提交Azure ML作业...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 解析作业名称
        import json
        job_info = json.loads(result.stdout)
        job_name = job_info.get("name", "unknown")
        
        print(f"✅ 简化DriveMM作业提交成功!")
        print(f"🆔 作业名称: {job_name}")
        print(f"🔗 监控链接: https://ml.azure.com/experiments/drivemm_simple_analysis/runs/{job_name}")
        
        return job_name
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 作业提交失败: {e}")
        print(f"错误输出: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return None

def main():
    """主函数"""
    print("🎯 DRIVEMM 简化版 AZURE A100 分析")
    print("=" * 50)
    print("🤖 使用简化的DriveMM分析逻辑")
    print("💎 在Azure A100 GPU上运行")
    print("📊 分析DADA-2000视频数据")
    print("🚨 检测鬼探头危险行为")
    print("=" * 50)
    
    # 提交简化分析作业
    job_name = submit_simple_job()
    
    if job_name:
        print("\n🎉 简化DriveMM分析作业启动成功!")
        print(f"📋 作业名称: {job_name}")
        print("📊 Azure ML Studio监控: https://ml.azure.com/")
        print("\n📝 下一步:")
        print("  1. 在Azure ML Studio中监控作业进度")
        print("  2. 等待作业完成(预计5-10分钟)")
        print("  3. 下载DriveMM分析结果")
        return 0
    else:
        print("\n❌ 简化DriveMM作业提交失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)