#!/usr/bin/env python3
"""
DriveMM公平比较Azure作业提交脚本
"""

import os
import sys
import subprocess
from datetime import datetime

def create_fair_comparison_job():
    """创建DriveMM公平比较作业配置"""
    
    job_yaml = f"""$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
code: ./
command: python azure_drivemm_fair_comparison.py
environment: azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10
compute: drivemm-a100-cluster
experiment_name: drivemm_fair_comparison
display_name: DriveMM_Fair_Comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}
description: Fair comparison of DriveMM using same prompts as GPT-4o and Gemini models
tags:
  model: DriveMM-Fair-Comparison
  dataset: DADA-2000
  task: ghost_probing_detection
  gpu: A100-80GB
  mode: fair_comparison
  prompt: balanced_gpt41_compatible
"""
    
    with open("drivemm_fair_comparison_job.yml", "w") as f:
        f.write(job_yaml)
    
    return "drivemm_fair_comparison_job.yml"

def submit_fair_comparison_job():
    """提交DriveMM公平比较作业"""
    print("🚀 提交DriveMM公平比较Azure A100作业...")
    print("📋 使用与GPT-4o和Gemini相同的平衡版prompt")
    print("=" * 60)
    
    try:
        # 创建作业配置
        job_file = create_fair_comparison_job()
        print(f"✅ 作业配置文件: {job_file}")
        
        # 检查脚本是否存在
        if not os.path.exists("azure_drivemm_fair_comparison.py"):
            print("❌ 公平比较脚本不存在")
            return None
        
        print(f"✅ 处理器脚本: azure_drivemm_fair_comparison.py")
        
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
        
        print(f"✅ DriveMM公平比较作业提交成功!")
        print(f"🆔 作业名称: {job_name}")
        print(f"🔗 监控链接: https://ml.azure.com/experiments/drivemm_fair_comparison/runs/{job_name}")
        
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
    print("🎯 DRIVEMM 公平比较 AZURE A100 分析")
    print("=" * 50)
    print("🤖 使用与GPT-4o和Gemini相同的平衡版prompt")
    print("📊 确保公平的模型比较实验")
    print("💎 在Azure A100 GPU上运行")
    print("🚨 标准化鬼探头检测criteria")
    print("=" * 50)
    
    # 提交公平比较作业
    job_name = submit_fair_comparison_job()
    
    if job_name:
        print("\n🎉 DriveMM公平比较作业启动成功!")
        print(f"📋 作业名称: {job_name}")
        print("📊 Azure ML Studio监控: https://ml.azure.com/")
        print("\n📝 下一步:")
        print("  1. 在Azure ML Studio中监控作业进度")
        print("  2. 等待作业完成(预计5-8分钟)")
        print("  3. 下载公平比较结果")
        print("  4. 与GPT-4o和Gemini结果进行对比")
        print("\n🔧 公平比较特性:")
        print("  - 使用相同的平衡版prompt结构")
        print("  - 三层检测机制(高确信度/潜在/正常)")
        print("  - 统一的JSON输出格式")
        print("  - 环境上下文理解")
        print("  - 标准化评判标准")
        return 0
    else:
        print("\n❌ DriveMM公平比较作业提交失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)