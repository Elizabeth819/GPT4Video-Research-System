#!/usr/bin/env python3
"""
调试版视频帧提取 - 详细日志
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
import logging
import torch
from typing import List, Dict, Optional, Tuple
import numpy as np
import time

# 设置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def extract_frames_with_detailed_logging(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """使用decord提取视频帧 - 详细日志版本"""
    
    logger.info(f"🎬 开始处理视频: {Path(video_path).name}")
    overall_start = time.time()
    
    try:
        import decord
        from decord import VideoReader
        
        # 设置decord使用native bridge
        decord.bridge.set_bridge('native')
        
        # 读取视频
        logger.debug(f"📂 视频路径: {video_path}")
        logger.debug(f"📂 文件存在: {Path(video_path).exists()}")
        logger.debug(f"📂 文件大小: {Path(video_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        video_load_start = time.time()
        video_reader = VideoReader(str(video_path))
        video_load_time = time.time() - video_load_start
        
        total_frames = len(video_reader)
        logger.info(f"📊 视频加载时间: {video_load_time:.4f}秒")
        logger.info(f"📊 总帧数: {total_frames}")
        
        if total_frames == 0:
            raise ValueError(f"视频文件没有帧: {video_path}")
        
        # 获取视频信息
        try:
            fps = video_reader.get_avg_fps()
            duration = total_frames / fps if fps > 0 else 0
            logger.info(f"📊 帧率: {fps:.2f} fps")
            logger.info(f"📊 时长: {duration:.2f}秒")
        except:
            logger.warning("⚠️  无法获取视频帧率信息")
        
        # 均匀分布选择帧
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        logger.info(f"📊 选择帧索引: {frame_indices.tolist()}")
        
        # 提取帧
        frames = []
        extraction_start = time.time()
        
        for i, idx in enumerate(frame_indices):
            frame_start = time.time()
            logger.debug(f"🔍 正在提取第 {i+1}/{num_frames} 帧 (索引: {idx})")
            
            # 获取帧
            frame_read_start = time.time()
            frame = video_reader[idx]
            frame_read_time = time.time() - frame_read_start
            
            logger.debug(f"  📖 帧读取时间: {frame_read_time:.4f}秒")
            logger.debug(f"  📖 帧类型: {type(frame)}")
            logger.debug(f"  📖 帧shape: {getattr(frame, 'shape', 'N/A')}")
            
            # 转换为numpy数组
            convert_start = time.time()
            if hasattr(frame, 'asnumpy'):
                frame_array = frame.asnumpy()
                logger.debug(f"  🔄 使用asnumpy()转换")
            elif isinstance(frame, torch.Tensor):
                frame_array = frame.cpu().numpy()
                logger.debug(f"  🔄 使用tensor.cpu().numpy()转换")
            else:
                frame_array = np.array(frame)
                logger.debug(f"  🔄 使用np.array()转换")
            
            convert_time = time.time() - convert_start
            logger.debug(f"  🔄 数组转换时间: {convert_time:.4f}秒")
            logger.debug(f"  🔄 转换后shape: {frame_array.shape}")
            logger.debug(f"  🔄 转换后dtype: {frame_array.dtype}")
            
            # 转换为PIL Image
            pil_start = time.time()
            pil_image = Image.fromarray(frame_array.astype(np.uint8))
            pil_time = time.time() - pil_start
            
            logger.debug(f"  🖼️  PIL转换时间: {pil_time:.4f}秒")
            logger.debug(f"  🖼️  PIL图像大小: {pil_image.size}")
            logger.debug(f"  🖼️  PIL图像模式: {pil_image.mode}")
            
            frames.append(pil_image)
            
            frame_total_time = time.time() - frame_start
            logger.info(f"  ✅ 帧 {i+1} 处理完成: {frame_total_time:.4f}秒")
        
        extraction_time = time.time() - extraction_start
        overall_time = time.time() - overall_start
        
        logger.info(f"✅ 所有帧提取时间: {extraction_time:.4f}秒")
        logger.info(f"✅ 总处理时间: {overall_time:.4f}秒")
        logger.info(f"✅ 成功提取 {len(frames)} 帧")
        
        return frames
        
    except Exception as e:
        overall_time = time.time() - overall_start
        logger.error(f"❌ 视频帧提取失败 ({overall_time:.4f}秒): {e}")
        raise

def main():
    """测试函数"""
    print("🔍 调试版视频帧提取测试")
    print("=" * 60)
    
    # 获取视频数据路径（模拟Azure ML环境）
    test_videos = [
        "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/images_1_001.avi",
        "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/images_1_002.avi"
    ]
    
    for video_path in test_videos:
        if Path(video_path).exists():
            try:
                frames = extract_frames_with_detailed_logging(video_path)
                print(f"🎉 成功提取 {len(frames)} 帧")
            except Exception as e:
                print(f"❌ 失败: {e}")
            print("-" * 40)
        else:
            print(f"❌ 文件不存在: {video_path}")

if __name__ == "__main__":
    main()