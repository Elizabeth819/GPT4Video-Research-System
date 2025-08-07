#!/usr/bin/env python3
"""
测试视频帧提取时间
"""
import time
import numpy as np
from PIL import Image
from pathlib import Path

def test_decord_extraction(video_path: str, num_frames: int = 8):
    """测试decord抽帧时间"""
    print(f"🎬 测试视频: {Path(video_path).name}")
    
    start_time = time.time()
    
    try:
        import decord
        from decord import VideoReader
        
        # 设置decord使用native bridge
        decord.bridge.set_bridge('native')
        
        load_start = time.time()
        # 读取视频
        video_reader = VideoReader(str(video_path))
        total_frames = len(video_reader)
        load_time = time.time() - load_start
        
        print(f"📊 视频加载时间: {load_time:.3f}秒")
        print(f"📊 总帧数: {total_frames}")
        
        if total_frames == 0:
            raise ValueError(f"视频文件没有帧: {video_path}")
        
        # 均匀分布选择帧
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        print(f"📊 选择帧索引: {frame_indices}")
        
        # 提取帧
        extract_start = time.time()
        frames = []
        for i, idx in enumerate(frame_indices):
            frame_start = time.time()
            frame = video_reader[idx]
            
            # 转换为numpy数组
            if hasattr(frame, 'asnumpy'):
                frame_array = frame.asnumpy()
            elif hasattr(frame, 'cpu'):
                frame_array = frame.cpu().numpy()
            else:
                frame_array = np.array(frame)
            
            # 转换为PIL Image
            pil_image = Image.fromarray(frame_array.astype(np.uint8))
            frames.append(pil_image)
            
            frame_time = time.time() - frame_start
            print(f"  帧 {i+1}/{num_frames} (索引{idx}): {frame_time:.3f}秒, 形状: {frame_array.shape}")
        
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
            frames, processing_time = test_decord_extraction(str(video_path))
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