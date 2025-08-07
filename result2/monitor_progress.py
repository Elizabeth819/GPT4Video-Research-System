#!/usr/bin/env python3
"""
Run8-Rerun实验进度监控脚本
"""

import os
import json
import glob
from datetime import datetime

def check_experiment_progress(experiment_name, base_dir):
    """检查实验进度"""
    print(f"\n📊 {experiment_name} 进度检查:")
    print("=" * 50)
    
    results_dirs = glob.glob(os.path.join(base_dir, "run8_rerun_plus_image_results_*"))
    
    if not results_dirs:
        print("❌ 未找到结果目录")
        return
    
    # 获取最新的结果目录
    latest_dir = max(results_dirs, key=os.path.getmtime)
    print(f"📁 结果目录: {os.path.basename(latest_dir)}")
    
    # 检查JSON结果文件
    json_files = glob.glob(os.path.join(latest_dir, "actionSummary_*.json"))
    print(f"✅ 已完成视频: {len(json_files)}/100")
    
    # 检查实验摘要
    summary_files = glob.glob(os.path.join(latest_dir, "experiment_summary_*.json"))
    if summary_files:
        summary_file = summary_files[0]
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            successful = summary.get("successful_analyses", 0)
            failed = summary.get("failed_analyses", 0)
            total_processed = len(summary.get("processed_videos", []))
            
            print(f"📈 成功分析: {successful}")
            print(f"❌ 失败分析: {failed}")
            print(f"📊 总处理数: {total_processed}")
            
            if summary.get("processing_errors"):
                print("⚠️  失败的视频:")
                for error in summary["processing_errors"]:
                    print(f"   - {error['video_id']}: {error.get('error_type', 'unknown')}")
                    
        except Exception as e:
            print(f"❌ 读取摘要文件失败: {e}")
    
    # 检查日志文件
    log_files = glob.glob(os.path.join(latest_dir, "*.log"))
    if log_files:
        log_file = log_files[0]
        try:
            # 读取最后几行日志
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    print(f"📝 最新日志 ({os.path.basename(log_file)}):")
                    for line in lines[-3:]:
                        print(f"   {line.strip()}")
        except Exception as e:
            print(f"❌ 读取日志文件失败: {e}")

def main():
    print("🔍 Run8-Rerun实验进度监控")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    base_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2"
    
    # 检查Run8-Rerun1
    run1_dir = os.path.join(base_path, "run8-rerun1")
    if os.path.exists(run1_dir):
        check_experiment_progress("Run8-Rerun1", run1_dir)
    else:
        print("\n❌ Run8-Rerun1目录不存在")
    
    # 检查Run8-Rerun2  
    run2_dir = os.path.join(base_path, "run8-rerun2")
    if os.path.exists(run2_dir):
        check_experiment_progress("Run8-Rerun2", run2_dir)
    else:
        print("\n❌ Run8-Rerun2目录不存在")
    
    print("\n" + "=" * 60)
    print("💡 提示:")
    print("- 每个视频大约需要25-30秒处理")
    print("- 100个视频预计需要45-60分钟")
    print("- 使用 'python result2/monitor_progress.py' 重新检查进度")

if __name__ == "__main__":
    main()