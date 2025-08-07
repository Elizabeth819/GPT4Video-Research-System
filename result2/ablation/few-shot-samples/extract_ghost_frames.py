#!/usr/bin/env python3
"""
鬼探头图像序列提取器
Extract ghost probing image sequences for multimodal few-shot learning
"""

import os
import sys
import cv2
from moviepy.editor import VideoFileClip

def extract_ghost_probing_frames():
    print("🎯 鬼探头图像序列提取器")
    print("=" * 50)
    
    # 配置路径
    project_root = "/Users/wanmeng/repository/GPT4Video-cobra-auto"
    video_dir = os.path.join(project_root, "DADA-2000-videos")
    output_dir = os.path.join(project_root, "result2/ablation/few-shot-samples/ghost_probing_images")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 视频配置
    video_configs = [
        {
            "file": "images_1_003.avi",
            "event_time": 2.0,
            "sample_id": "sample1",
            "description": "行人从黑色车后突然出现"
        },
        {
            "file": "images_1_006.avi", 
            "event_time": 6.0,
            "sample_id": "sample2",
            "description": "多个行人从车辆后突然出现"
        },
        {
            "file": "images_1_008.avi",
            "event_time": 2.0,
            "sample_id": "sample3", 
            "description": "行人从白色卡车后突然出现"
        }
    ]
    
    extracted_files = []
    
    for config in video_configs:
        video_file = config["file"]
        event_time = config["event_time"]
        sample_id = config["sample_id"]
        description = config["description"]
        
        video_path = os.path.join(video_dir, video_file)
        
        print(f"\n📹 处理视频: {video_file}")
        print(f"   描述: {description}")
        print(f"   事件时间: {event_time}s")
        
        if not os.path.exists(video_path):
            print(f"   ❌ 视频文件不存在: {video_path}")
            continue
            
        try:
            # 加载视频
            with VideoFileClip(video_path) as clip:
                print(f"   ✅ 视频加载成功，时长: {clip.duration:.2f}s")
                
                # 定义三个关键帧时间点
                frames_to_extract = [
                    {
                        "timestamp": event_time - 0.5,
                        "phase": "before",
                        "description": "正常场景，行人被遮挡物隐藏"
                    },
                    {
                        "timestamp": event_time,
                        "phase": "during", 
                        "description": "行人正在从遮挡物后出现"
                    },
                    {
                        "timestamp": event_time + 0.5,
                        "phase": "after",
                        "description": "行人已出现在车辆路径中"
                    }
                ]
                
                # 提取每个关键帧
                for frame_config in frames_to_extract:
                    timestamp = frame_config["timestamp"]
                    phase = frame_config["phase"]
                    frame_desc = frame_config["description"]
                    
                    # 检查时间戳是否在视频范围内
                    if timestamp < 0 or timestamp > clip.duration:
                        print(f"   ⚠️ {phase}帧时间戳{timestamp:.1f}s超出视频范围")
                        continue
                    
                    try:
                        # 提取帧
                        frame = clip.get_frame(timestamp)
                        
                        # 转换颜色格式 (RGB -> BGR for OpenCV)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # 生成文件名
                        filename = f"ghost_probing_{sample_id}_{phase}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        # 保存图像
                        success = cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        if success:
                            file_size = os.path.getsize(filepath)
                            print(f"   ✅ {phase}帧: {filename} ({file_size:,} bytes)")
                            print(f"      时间: {timestamp:.1f}s - {frame_desc}")
                            extracted_files.append(filename)
                        else:
                            print(f"   ❌ {phase}帧保存失败")
                            
                    except Exception as e:
                        print(f"   ❌ {phase}帧提取失败: {str(e)}")
                        
        except Exception as e:
            print(f"   ❌ 视频处理失败: {str(e)}")
    
    # 输出结果总结
    print("\n" + "=" * 50)
    print(f"🎉 提取完成！共提取 {len(extracted_files)} 个图像文件")
    
    if extracted_files:
        print(f"\n📁 输出目录: {output_dir}")
        print("\n📋 提取的文件列表:")
        
        # 按样本分组显示
        for i in range(1, 4):
            sample_files = [f for f in extracted_files if f"sample{i}" in f]
            if sample_files:
                print(f"\n   Sample {i} (对应 {video_configs[i-1]['file']}):")
                for filename in sorted(sample_files):
                    filepath = os.path.join(output_dir, filename)
                    if os.path.exists(filepath):
                        size = os.path.getsize(filepath)
                        print(f"     {filename} ({size:,} bytes)")
        
        print("\n🔍 图像序列说明:")
        print("   • BEFORE: 正常驾驶场景，行人被遮挡物隐藏")
        print("   • DURING: 关键时刻，行人正从遮挡物后出现") 
        print("   • AFTER:  危险情况，行人已出现在车辆路径中")
        
        print("\n💡 用途:")
        print("   这些图像序列展示了完整的鬼探头过程，")
        print("   可用于多模态few-shot学习，相比纯文本描述")
        print("   能提供更丰富的视觉空间关系信息。")
        
        return True
    else:
        print("❌ 未成功提取任何图像文件")
        return False

if __name__ == "__main__":
    try:
        success = extract_ghost_probing_frames()
        if success:
            print("\n✅ 鬼探头图像序列提取成功完成！")
        else:
            print("\n❌ 鬼探头图像序列提取失败")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 程序执行错误: {str(e)}")
        sys.exit(1)