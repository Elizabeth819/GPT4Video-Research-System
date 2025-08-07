#!/usr/bin/env python3
"""
Progress checker for Run 18
"""
import os
from pathlib import Path
import json
import datetime

def check_progress():
    output_dir = Path(__file__).parent
    
    # Count processed files
    processed_files = list(output_dir.glob("actionSummary_*.json"))
    processed_count = len(processed_files)
    
    # Total DADA-100 videos (typically 101)
    total_videos = 101
    
    # Calculate progress
    progress_percent = (processed_count / total_videos) * 100
    
    print(f"🚀 Run 18 Progress Check - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Processed: {processed_count}/{total_videos} videos ({progress_percent:.1f}%)")
    print(f"⏱️  Remaining: {total_videos - processed_count} videos")
    
    if processed_count > 0:
        # Show last few processed videos
        processed_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"\n📋 Last 5 processed videos:")
        for i, file in enumerate(processed_files[:5]):
            video_name = file.stem.replace("actionSummary_", "")
            mod_time = datetime.datetime.fromtimestamp(file.stat().st_mtime)
            print(f"   {i+1}. {video_name} - {mod_time.strftime('%H:%M:%S')}")
    
    # Check for log files
    log_files = list(output_dir.glob("logs/run18_*.log"))
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        print(f"\n📄 Latest log: {latest_log.name}")
        
        # Show last few lines of log
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 3:
                    print("   Last 3 log entries:")
                    for line in lines[-3:]:
                        if line.strip():
                            print(f"   {line.strip()}")
        except Exception as e:
            print(f"   Could not read log: {e}")

if __name__ == "__main__":
    check_progress()