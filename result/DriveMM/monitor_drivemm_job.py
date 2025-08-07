#!/usr/bin/env python3
"""
监控DriveMM作业状态脚本
定期检查作业状态并显示进度
"""

import time
import subprocess
import json
import sys
from datetime import datetime

def run_command(command):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def get_job_status(job_name):
    """获取作业状态"""
    command = f"az ml job show --name {job_name} --resource-group drivelm-rg --workspace-name drivelm-ml-workspace --query '{{Name:name,Status:status,StartTime:creation_context.created_at}}' --output json"
    
    stdout, stderr, returncode = run_command(command)
    
    if returncode == 0:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return None
    else:
        print(f"❌ 获取作业状态失败: {stderr}")
        return None

def monitor_job(job_name, check_interval=60):
    """监控作业状态"""
    print(f"🔍 开始监控作业: {job_name}")
    print(f"📊 检查间隔: {check_interval}秒")
    print("=" * 60)
    
    last_status = None
    start_time = datetime.now()
    
    while True:
        current_time = datetime.now()
        elapsed = current_time - start_time
        
        job_info = get_job_status(job_name)
        
        if job_info:
            status = job_info.get('Status', 'Unknown')
            
            if status != last_status:
                print(f"⏰ [{current_time.strftime('%H:%M:%S')}] 状态变更: {status}")
                last_status = status
            
            if status == "Completed":
                print("✅ 作业已完成!")
                print(f"💾 总耗时: {elapsed}")
                break
            elif status == "Failed":
                print("❌ 作业失败!")
                print(f"💾 总耗时: {elapsed}")
                break
            elif status == "Canceled":
                print("🔴 作业已取消!")
                print(f"💾 总耗时: {elapsed}")
                break
            elif status == "Running":
                print(f"🔄 [{current_time.strftime('%H:%M:%S')}] 作业运行中... (已运行 {elapsed})")
            else:
                print(f"⏳ [{current_time.strftime('%H:%M:%S')}] 状态: {status} (已等待 {elapsed})")
        else:
            print(f"❌ [{current_time.strftime('%H:%M:%S')}] 无法获取作业状态")
        
        time.sleep(check_interval)

def main():
    """主函数"""
    job_name = "red_diamond_xfbmkt8klp"
    
    print("🚀 DriveMM作业监控器")
    print("=" * 60)
    print(f"📋 作业名称: {job_name}")
    print(f"🔗 监控URL: https://ml.azure.com/runs/{job_name}?wsid=/subscriptions/0d3f39ba-7349-4bd7-8122-649ff18f0a4a/resourcegroups/drivelm-rg/workspaces/drivelm-ml-workspace&tid=16b3c013-d300-468d-ac64-7eda0820b6d3")
    print("=" * 60)
    
    try:
        monitor_job(job_name)
    except KeyboardInterrupt:
        print("\n🔴 监控已停止")
        sys.exit(0)

if __name__ == "__main__":
    main()