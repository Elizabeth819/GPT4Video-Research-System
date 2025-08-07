#!/usr/bin/env python3
import os
import cv2
from moviepy.editor import VideoFileClip

# 配置
video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/ghost_probing_images"
os.makedirs(output_dir, exist_ok=True)

# 提取配置
extractions = [
    ("images_1_003.avi", 2.0, "sample1"),
    ("images_1_006.avi", 6.0, "sample2"), 
    ("images_1_008.avi", 2.0, "sample3")
]

print("开始提取鬼探头图像序列...")

for video_file, event_time, sample_id in extractions:
    video_path = os.path.join(video_dir, video_file)
    print(f"处理: {video_file} (事件时间: {event_time}s)")
    
    if os.path.exists(video_path):
        try:
            clip = VideoFileClip(video_path)
            
            # 提取3个时间点的帧
            for delta, phase in [(-0.5, "before"), (0, "during"), (0.5, "after")]:
                timestamp = event_time + delta
                if 0 <= timestamp <= clip.duration:
                    frame = clip.get_frame(timestamp)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    filename = f"ghost_probing_{sample_id}_{phase}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"  ✓ {filename}")
            
            clip.close()
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    else:
        print(f"  ✗ 文件不存在: {video_path}")

print(f"完成! 图像保存在: {output_dir}")