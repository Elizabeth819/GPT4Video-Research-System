#!/usr/bin/env python3
"""
Monitor Run 16 Progress
实时监控Run 16的处理进度
"""

import os
import time
from pathlib import Path
import json

def monitor_run16():
    """监控Run 16进度"""
    run16_dir = Path(__file__).parent
    log_dir = run16_dir / "logs"
    
    while True:
        try:
            # 统计已完成的结果文件
            result_files = list(run16_dir.glob("actionSummary_images_*.json"))
            completed_count = len(result_files)
            
            # 统计ghost probing检测结果
            ghost_probing_count = 0
            for result_file in result_files:
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    key_actions = result.get('key_actions', '').lower()
                    if 'ghost probing' in key_actions:
                        ghost_probing_count += 1
                except:
                    continue
            
            # 检测率
            detection_rate = (ghost_probing_count / completed_count * 100) if completed_count > 0 else 0
            
            # 获取最新日志
            latest_log = None
            if log_dir.exists():
                log_files = list(log_dir.glob("run16_gemini_2_5_flash_fewshot_*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            
            # 显示进度
            print(f"\r🚀 Run 16 Progress: {completed_count}/100 videos processed "
                  f"| Ghost Probing: {ghost_probing_count} ({detection_rate:.1f}%)", end="")
            
            # 如果有日志文件，显示最近的错误或重要信息
            if latest_log and latest_log.exists():
                try:
                    with open(latest_log, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # 查找最后几行的重要信息
                    recent_errors = []
                    recent_success = []
                    for line in lines[-10:]:
                        if "ERROR" in line:
                            recent_errors.append(line.strip())
                        elif "Successfully analyzed" in line:
                            recent_success.append(line.strip())
                    
                    if recent_errors:
                        print(f"\n⚠️  Recent Error: {recent_errors[-1]}")
                    elif recent_success:
                        print(f"\n✅ Latest: {recent_success[-1]}")
                except:
                    pass
            
            # 每10秒刷新一次
            time.sleep(10)
            
        except KeyboardInterrupt:
            print(f"\n\n📊 Final Status: {completed_count}/100 videos completed")
            print(f"🎯 Ghost Probing Detection: {ghost_probing_count} cases ({detection_rate:.1f}%)")
            break
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    print("🔍 Starting Run 16 Monitor...")
    print("Press Ctrl+C to stop monitoring")
    monitor_run16()