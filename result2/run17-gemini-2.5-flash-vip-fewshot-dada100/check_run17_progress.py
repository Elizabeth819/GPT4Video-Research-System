#!/usr/bin/env python3
"""
Progress checker for Run 17
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
    
    # Total DADA-100 videos (we started from video 6, so need to process 96 videos)
    total_videos = 96  # From 6th video to 101st
    original_processed = 4  # Already had 4 videos processed
    
    # Calculate progress
    progress_percent = (processed_count / (total_videos + original_processed)) * 100
    
    print(f"🚀 Run 17 Progress Check - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Processed: {processed_count}/101 videos ({progress_percent:.1f}%)")
    print(f"⏱️  Remaining: {101 - processed_count} videos")
    print(f"🔄 Currently processing: {processed_count - original_processed}/96 additional videos")
    
    if processed_count > 0:
        # Show last few processed videos
        processed_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"\n📋 Last 5 processed videos:")
        for i, file in enumerate(processed_files[:5]):
            video_name = file.stem.replace("actionSummary_", "")
            mod_time = datetime.datetime.fromtimestamp(file.stat().st_mtime)
            print(f"   {i+1}. {video_name} - {mod_time.strftime('%H:%M:%S')}")
    
    # Estimate completion time
    if processed_count > original_processed:
        avg_time_per_video = 12  # Based on log analysis
        remaining_videos = 101 - processed_count
        estimated_minutes = (remaining_videos * avg_time_per_video) / 60
        
        completion_time = datetime.datetime.now() + datetime.timedelta(minutes=estimated_minutes)
        print(f"\n⏰ Estimated completion: {completion_time.strftime('%H:%M:%S')} ({estimated_minutes:.1f} minutes)")
    
    # Check processing rate
    print(f"\n📈 Processing rate: ~12 seconds/video average")
    print(f"🎯 Model: Gemini-2.5-Flash + VIP详细prompt + Few-shot Examples")

if __name__ == "__main__":
    check_progress()