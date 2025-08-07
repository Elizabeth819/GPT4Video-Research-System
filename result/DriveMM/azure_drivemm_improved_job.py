#!/usr/bin/env python3
"""
改进版DriveMM Azure A100作业提交脚本
"""

import os
import sys
import subprocess
from datetime import datetime

def create_improved_job():
    """创建改进的DriveMM作业配置"""
    
    job_yaml = f"""$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
code: ./
command: python azure_real_drivemm_improved.py
environment: azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10
compute: drivemm-a100-cluster
experiment_name: drivemm_improved_analysis
display_name: DriveMM_Improved_DADA2000_{datetime.now().strftime('%Y%m%d_%H%M%S')}
description: Improved DriveMM analysis with advanced video feature extraction on DADA-2000
tags:
  model: DriveMM-Improved
  dataset: DADA-2000
  task: ghost_probing_detection
  gpu: A100-80GB
  mode: advanced_analysis
"""
    
    with open("drivemm_improved_job.yml", "w") as f:
        f.write(job_yaml)
    
    return "drivemm_improved_job.yml"

def submit_improved_job():
    """提交改进的DriveMM作业"""
    print("🚀 提交改进版DriveMM Azure A100作业...")
    
    try:
        # 创建作业配置
        job_file = create_improved_job()
        print(f"✅ 作业配置文件: {job_file}")
        
        # 检查改进脚本是否存在
        if not os.path.exists("azure_real_drivemm_improved.py"):
            print("❌ 改进脚本不存在")
            return None
        
        print(f"✅ 处理器脚本: azure_real_drivemm_improved.py")
        
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
        
        print(f"✅ 改进DriveMM作业提交成功!")
        print(f"🆔 作业名称: {job_name}")
        print(f"🔗 监控链接: https://ml.azure.com/experiments/drivemm_improved_analysis/runs/{job_name}")
        
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
    print("🎯 DRIVEMM 改进版 AZURE A100 分析")
    print("=" * 50)
    print("🤖 使用改进的DriveMM分析逻辑")
    print("🔍 包含视频特征提取和概率分析")
    print("💎 在Azure A100 GPU上运行")
    print("📊 分析DADA-2000视频数据")
    print("🚨 智能鬼探头检测")
    print("=" * 50)
    
    # 提交改进分析作业
    job_name = submit_improved_job()
    
    if job_name:
        print("\n🎉 改进DriveMM分析作业启动成功!")
        print(f"📋 作业名称: {job_name}")
        print("📊 Azure ML Studio监控: https://ml.azure.com/")
        print("\n📝 下一步:")
        print("  1. 在Azure ML Studio中监控作业进度")
        print("  2. 等待作业完成(预计8-12分钟)")
        print("  3. 下载改进的DriveMM分析结果")
        print("\n🔧 改进功能:")
        print("  - 视频特征提取(复杂度、运动强度)")
        print("  - 概率化风险评估")
        print("  - 多因素综合分析")
        print("  - 详细的分析报告")
        return 0
    else:
        print("\n❌ 改进DriveMM作业提交失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)