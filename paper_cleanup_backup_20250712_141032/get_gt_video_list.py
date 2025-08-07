#!/usr/bin/env python3
"""
获取Ground Truth覆盖的视频列表
"""

import csv
import os

def get_ground_truth_videos():
    """获取Ground Truth标签中涉及的所有视频"""
    ground_truth_path = "result/groundtruth_labels.csv"
    gt_videos = []
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['video_id'] and row['video_id'].endswith('.avi'):
                video_id = row['video_id'].replace('.avi', '')
                gt_videos.append(video_id)
    
    return sorted(gt_videos)

def check_video_availability(gt_videos):
    """检查哪些Ground Truth视频在DADA-2000-videos文件夹中可用"""
    video_folder = "DADA-2000-videos"
    available_videos = []
    missing_videos = []
    
    for video_id in gt_videos:
        video_path = os.path.join(video_folder, f"{video_id}.avi")
        if os.path.exists(video_path):
            available_videos.append(video_id)
        else:
            missing_videos.append(video_id)
    
    return available_videos, missing_videos

def main():
    print("🔍 分析Ground Truth视频覆盖范围")
    print("=" * 50)
    
    # 获取Ground Truth视频列表
    gt_videos = get_ground_truth_videos()
    print(f"📁 Ground Truth标签中的视频总数: {len(gt_videos)}")
    
    # 按系列分组
    series_groups = {}
    for video_id in gt_videos:
        if video_id.startswith('images_'):
            parts = video_id.split('_')
            if len(parts) >= 3:
                series = f"images_{parts[1]}_XXX"
                if series not in series_groups:
                    series_groups[series] = []
                series_groups[series].append(video_id)
    
    print(f"\n📊 按系列分组:")
    for series, videos in sorted(series_groups.items()):
        print(f"   {series}: {len(videos)} 个视频")
        print(f"      范围: {min(videos)} - {max(videos)}")
    
    # 检查视频可用性
    available_videos, missing_videos = check_video_availability(gt_videos)
    print(f"\n✅ 可用视频: {len(available_videos)}")
    print(f"❌ 缺失视频: {len(missing_videos)}")
    
    if missing_videos:
        print(f"\n缺失的视频:")
        for video in missing_videos[:10]:  # 只显示前10个
            print(f"   {video}")
        if len(missing_videos) > 10:
            print(f"   ... 还有 {len(missing_videos) - 10} 个")
    
    # 输出可用视频列表（用于批处理）
    print(f"\n💾 保存可用Ground Truth视频列表...")
    with open("gt_available_videos.txt", 'w') as f:
        for video in available_videos:
            f.write(f"{video}.avi\n")
    
    print(f"✅ 可用视频列表保存到: gt_available_videos.txt")
    
    return available_videos

if __name__ == "__main__":
    available_videos = main()