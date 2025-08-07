#!/usr/bin/env python3
"""
Run 13 监控脚本
实时查看分析进度和结果统计
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

class Run13Monitor:
    def __init__(self):
        self.output_dir = Path(__file__).parent
        self.dada_100_dir = Path("/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos")
        
    def get_progress(self):
        """获取当前进度"""
        # 统计视频总数
        total_videos = len(list(self.dada_100_dir.glob("images_*.avi")))
        
        # 统计已处理的结果文件
        processed_files = list(self.output_dir.glob("actionSummary_images_*.json"))
        processed_count = len(processed_files)
        
        # 统计ghost probing检测结果
        ghost_probing_count = 0
        cut_in_count = 0
        none_count = 0
        
        for result_file in processed_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    key_actions = result.get('key_actions', '').lower()
                    
                    if 'ghost probing' in key_actions:
                        ghost_probing_count += 1
                    elif 'cut-in' in key_actions:
                        cut_in_count += 1
                    elif key_actions in ['none', '']:
                        none_count += 1
            except Exception as e:
                continue
        
        return {
            'total_videos': total_videos,
            'processed_count': processed_count,
            'remaining_count': total_videos - processed_count,
            'progress_percentage': (processed_count / total_videos * 100) if total_videos > 0 else 0,
            'ghost_probing_count': ghost_probing_count,
            'cut_in_count': cut_in_count,
            'none_count': none_count
        }
    
    def get_latest_log(self):
        """获取最新日志内容"""
        log_files = list(self.output_dir.glob("logs/run13_gemini_*.log"))
        if not log_files:
            return "No log files found"
        
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 返回最后10行
                return ''.join(lines[-10:])
        except Exception as e:
            return f"Error reading log: {e}"
    
    def show_status(self):
        """显示状态信息"""
        progress = self.get_progress()
        
        print("=" * 60)
        print("🔍 Run 13: Gemini 2.0 Flash Analysis - Status Monitor")
        print("=" * 60)
        print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 进度信息
        print("📊 Progress:")
        print(f"  Total Videos: {progress['total_videos']}")
        print(f"  Processed: {progress['processed_count']}")
        print(f"  Remaining: {progress['remaining_count']}")
        print(f"  Progress: {progress['progress_percentage']:.1f}%")
        print()
        
        # 结果统计
        if progress['processed_count'] > 0:
            print("🎯 Detection Results:")
            print(f"  Ghost Probing: {progress['ghost_probing_count']}")
            print(f"  Cut-in: {progress['cut_in_count']}")
            print(f"  None: {progress['none_count']}")
            print(f"  Ghost Probing Rate: {progress['ghost_probing_count']/progress['processed_count']*100:.1f}%")
            print()
        
        # 最新日志
        print("📝 Latest Log (last 10 lines):")
        print("-" * 40)
        print(self.get_latest_log())
        print("-" * 40)
        
    def monitor_continuously(self, interval=30):
        """持续监控"""
        print("🔄 Starting continuous monitoring (Ctrl+C to stop)")
        print(f"📱 Refresh interval: {interval} seconds")
        print()
        
        try:
            while True:
                os.system('clear')  # 清屏
                self.show_status()
                print(f"\n⏳ Next refresh in {interval} seconds... (Ctrl+C to stop)")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped by user")

def main():
    """主函数"""
    monitor = Run13Monitor()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        monitor.monitor_continuously(interval)
    else:
        monitor.show_status()
        print("\n💡 Use --continuous [interval] for continuous monitoring")
        print("   Example: python monitor_run13.py --continuous 60")

if __name__ == "__main__":
    main()