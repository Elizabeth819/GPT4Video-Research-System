#!/usr/bin/env python3
"""
使用OpenCV测试视频帧提取时间
"""
import time
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def test_opencv_extraction(video_path: str, num_frames: int = 8):
    """测试OpenCV抽帧时间"""
    print(f"🎬 测试视频: {Path(video_path).name}")
    
    start_time = time.time()
    
    try:
        # 打开视频
        load_start = time.time()
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        load_time = time.time() - load_start
        
        print(f"📊 视频加载时间: {load_time:.3f}秒")
        print(f"📊 总帧数: {total_frames}")
        print(f"📊 帧率: {fps:.2f} fps")
        print(f"📊 时长: {duration:.2f}秒")
        
        if total_frames == 0:
            raise ValueError(f"视频文件没有帧: {video_path}")
        
        # 均匀分布选择帧
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        print(f"📊 选择帧索引: {frame_indices}")
        
        # 提取帧
        extract_start = time.time()
        frames = []
        
        for i, frame_idx in enumerate(frame_indices):
            frame_start = time.time()
            
            # 跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"⚠️  无法读取帧 {frame_idx}")
                continue
            
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            
            frame_time = time.time() - frame_start
            print(f"  帧 {i+1}/{num_frames} (索引{frame_idx}): {frame_time:.3f}秒, 形状: {frame_rgb.shape}")
        
        cap.release()
        
        extract_time = time.time() - extract_start
        total_time = time.time() - start_time
        
        print(f"✅ 帧提取时间: {extract_time:.3f}秒")
        print(f"✅ 总处理时间: {total_time:.3f}秒")
        print(f"✅ 成功提取 {len(frames)} 帧")
        
        return frames, total_time
        
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        return None, 0

def main():
    """测试几个视频的抽帧时间"""
    video_folder = Path("/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos")
    
    # 测试前5个视频
    test_videos = [
        "images_1_001.avi",
        "images_1_002.avi", 
        "images_1_003.avi",
        "images_1_004.avi",
        "images_1_005.avi"
    ]
    
    total_times = []
    
    for video_name in test_videos:
        video_path = video_folder / video_name
        if video_path.exists():
            print("=" * 60)
            frames, processing_time = test_opencv_extraction(str(video_path))
            total_times.append(processing_time)
            print()
        else:
            print(f"❌ 视频文件不存在: {video_path}")
    
    if total_times:
        avg_time = sum(total_times) / len(total_times)
        print("=" * 60)
        print("📊 统计结果:")
        for i, t in enumerate(total_times):
            print(f"  视频 {i+1}: {t:.3f}秒")
        print(f"平均时间: {avg_time:.3f}秒")
        print("=" * 60)

if __name__ == "__main__":
    main()