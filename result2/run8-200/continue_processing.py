#!/usr/bin/env python3
"""
检查并继续处理剩余的DADA-200视频
确保完整处理200个视频
"""

import os
import json
import glob

def find_missing_videos():
    """找出未处理的视频"""
    
    # 获取所有200个视频文件
    video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/DADA-200-videos"
    all_videos = []
    for filename in os.listdir(video_dir):
        if filename.endswith('.avi'):
            video_id = filename.replace('.avi', '')
            all_videos.append(video_id)
    
    all_videos.sort()
    print(f"📂 DADA-200目录下总视频数: {len(all_videos)}")
    
    # 读取已处理的视频
    results_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-200/run8_200videos_final_results_20250730_134411.json"
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_videos = set()
        for result in data['detailed_results']:
            video_id = result['video_id'].replace('.avi', '')
            processed_videos.add(video_id)
        
        print(f"✅ 已处理的视频数: {len(processed_videos)}")
        
        # 找出未处理的视频
        missing_videos = []
        for video_id in all_videos:
            if video_id not in processed_videos:
                missing_videos.append(video_id)
        
        missing_videos.sort()
        print(f"❌ 未处理的视频数: {len(missing_videos)}")
        
        if missing_videos:
            print("\n📋 未处理的视频列表:")
            for i, video_id in enumerate(missing_videos, 1):
                print(f"   {i:2d}. {video_id}.avi")
        
        return missing_videos
        
    except Exception as e:
        print(f"❌ 读取结果文件失败: {e}")
        return []

def check_api_timeout_videos():
    """检查API超时失败的视频"""
    
    log_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-200/run8_200videos_output.log"
    
    timeout_videos = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 查找API超时的视频
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "API请求异常" in line and "Read timed out" in line:
                # 查找前面的开始分析日志
                for j in range(i-1, max(0, i-10), -1):
                    if "🎬 开始分析视频:" in lines[j]:
                        video_id = lines[j].split("🎬 开始分析视频: ")[1].strip()
                        timeout_videos.append(video_id)
                        break
        
        timeout_videos = list(set(timeout_videos))  # 去重
        print(f"\n⏰ API超时失败的视频数: {len(timeout_videos)}")
        if timeout_videos:
            print("📋 超时视频列表:")
            for i, video_id in enumerate(timeout_videos, 1):
                print(f"   {i:2d}. {video_id}.avi")
        
        return timeout_videos
        
    except Exception as e:
        print(f"❌ 读取日志文件失败: {e}")
        return []

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Run 8-200视频处理完整性检查")
    print("=" * 60)
    
    missing_videos = find_missing_videos()
    timeout_videos = check_api_timeout_videos()
    
    # 合并需要重新处理的视频
    videos_to_process = list(set(missing_videos + timeout_videos))
    videos_to_process.sort()
    
    print(f"\n🎯 需要(重新)处理的视频总数: {len(videos_to_process)}")
    
    if videos_to_process:
        print("\n📝 建议操作:")
        print("   1. 修改run8_gpt4o_200videos_fewshot.py，设置只处理这些视频")
        print("   2. 或者创建一个专门的补充处理脚本")
        print("   3. 处理完成后合并结果到主结果文件")
        
        # 保存待处理视频列表
        todo_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-200/videos_to_process.json"
        todo_data = {
            "missing_videos": missing_videos,
            "timeout_videos": timeout_videos, 
            "videos_to_process": videos_to_process,
            "total_count": len(videos_to_process)
        }
        
        with open(todo_file, 'w', encoding='utf-8') as f:
            json.dump(todo_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 待处理视频列表已保存到: {todo_file}")
    else:
        print("\n✅ 所有200个视频已完成处理！")