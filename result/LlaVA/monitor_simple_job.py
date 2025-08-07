#!/usr/bin/env python3
"""
监控简化版LLaVA作业
"""
import subprocess
import time
import json
from datetime import datetime

JOB_NAME = "khaki_cloud_c1y816xhw5"
RESOURCE_GROUP = "llava-resourcegroup"
WORKSPACE_NAME = "llava-workspace"

def get_job_status():
    """获取作业状态"""
    cmd = [
        "az", "ml", "job", "show", 
        "-n", JOB_NAME,
        "--resource-group", RESOURCE_GROUP,
        "--workspace-name", WORKSPACE_NAME,
        "--output", "json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return json.loads(result.stdout)
    return None

def monitor_job():
    """监控作业直到完成"""
    print(f"🔍 开始监控简化版LLaVA作业: {JOB_NAME}")
    print("=" * 60)
    
    start_time = datetime.now()
    last_status = None
    
    while True:
        job_info = get_job_status()
        if not job_info:
            print("❌ 无法获取作业信息")
            break
            
        status = job_info.get("status", "Unknown")
        display_name = job_info.get("display_name", "")
        
        if status != last_status:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 状态: {status} (已运行 {elapsed:.0f}秒)")
            last_status = status
            
            # 显示作业URL
            if "services" in job_info and "Studio" in job_info["services"]:
                print(f"🔗 作业URL: {job_info['services']['Studio']['endpoint']}")
        
        # 检查是否完成
        if status in ["Completed", "Failed", "Canceled"]:
            print(f"\n{'✅' if status == 'Completed' else '❌'} 作业{status}!")
            
            # 获取输出路径
            if "outputs" in job_info:
                print("\n📁 输出路径:")
                for name, output in job_info["outputs"].items():
                    if "path" in output:
                        print(f"  - {name}: {output['path']}")
            
            if status == "Completed":
                print("\n🎉 简化版LLaVA检测成功完成！")
                print("📥 准备下载结果...")
            break
            
        time.sleep(30)  # 每30秒检查一次
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  总运行时间: {total_time/60:.1f}分钟")
    
    return status == "Completed"

if __name__ == "__main__":
    success = monitor_job()
    
    if success:
        print("\n📥 下载作业结果...")
        download_cmd = [
            "az", "ml", "job", "download",
            "-n", JOB_NAME,
            "--all",
            "--resource-group", RESOURCE_GROUP,
            "--workspace-name", WORKSPACE_NAME,
            "-p", "simple_job_results"
        ]
        
        result = subprocess.run(download_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 结果下载成功到 simple_job_results 目录")
        else:
            print("❌ 结果下载失败")