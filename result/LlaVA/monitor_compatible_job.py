#!/usr/bin/env python3
"""
监控兼容版LLaVA作业
"""
import subprocess
import time
import json
from datetime import datetime

JOB_NAME = "coral_jewel_sz5cgqwbhl"
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

def get_job_logs():
    """获取作业日志片段"""
    cmd = [
        "az", "ml", "job", "stream",
        "-n", JOB_NAME,
        "--resource-group", RESOURCE_GROUP,
        "--workspace-name", WORKSPACE_NAME
    ]
    
    # 使用timeout来获取最新日志
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        return lines[-20:]  # 返回最后20行
    return []

def monitor_job():
    """监控作业直到完成"""
    print(f"🔍 开始监控兼容版LLaVA作业: {JOB_NAME}")
    print("=" * 60)
    
    start_time = datetime.now()
    last_status = None
    log_check_counter = 0
    
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
        
        # 每5次检查获取一次日志
        log_check_counter += 1
        if log_check_counter >= 5 and status == "Running":
            log_check_counter = 0
            print(f"\n📄 最新日志片段:")
            try:
                logs = get_job_logs()
                for line in logs[-10:]:  # 显示最后10行
                    if line.strip():
                        print(f"  {line}")
            except Exception as e:
                print(f"  ⚠️  获取日志失败: {e}")
        
        # 检查是否完成
        if status in ["Completed", "Failed", "Canceled"]:
            print(f"\n{'✅' if status == 'Completed' else '❌'} 作业{status}!")
            
            # 获取输出路径
            if "outputs" in job_info:
                print("\n📁 输出路径:")
                for name, output in job_info["outputs"].items():
                    if "path" in output:
                        print(f"  - {name}: {output['path']}")
            
            # 如果失败，尝试获取错误信息
            if status == "Failed":
                print("\n❌ 获取最后的日志信息...")
                try:
                    logs = get_job_logs()
                    print("最后日志:")
                    for line in logs:
                        print(f"  {line}")
                except:
                    pass
            break
            
        time.sleep(30)  # 每30秒检查一次
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  总运行时间: {total_time/60:.1f}分钟")

if __name__ == "__main__":
    monitor_job()