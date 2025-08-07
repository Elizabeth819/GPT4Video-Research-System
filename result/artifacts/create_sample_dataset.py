#!/usr/bin/env python3
"""
创建包含100个视频的样本数据集
"""

import os
import shutil
from pathlib import Path

def create_sample_dataset():
    """创建包含前100个视频的样本数据集"""
    
    source_dir = Path("./DADA-2000-videos")
    target_dir = Path("./DADA-100-videos")
    
    if not source_dir.exists():
        print(f"错误：源目录不存在: {source_dir}")
        return False
    
    # 创建目标目录
    target_dir.mkdir(exist_ok=True)
    
    # 获取前100个视频
    selected_videos = []
    for i in range(1, 6):  # 1 to 5
        pattern = f"images_{i}_*.avi"
        videos = list(source_dir.glob(pattern))
        selected_videos.extend(videos)
    
    # 排序并限制为100个
    selected_videos.sort()
    selected_videos = selected_videos[:100]
    
    print(f"准备复制 {len(selected_videos)} 个视频到 {target_dir}")
    
    # 复制视频文件
    for i, video in enumerate(selected_videos):
        target_path = target_dir / video.name
        
        if not target_path.exists():
            print(f"[{i+1:3d}/{len(selected_videos)}] 复制 {video.name}")
            shutil.copy2(video, target_path)
        else:
            print(f"[{i+1:3d}/{len(selected_videos)}] 跳过 {video.name} (已存在)")
    
    # 验证复制结果
    copied_videos = list(target_dir.glob("*.avi"))
    print(f"\n✅ 复制完成！")
    print(f"   - 源目录: {source_dir}")
    print(f"   - 目标目录: {target_dir}")
    print(f"   - 复制的视频数: {len(copied_videos)}")
    
    # 计算目录大小
    total_size = sum(f.stat().st_size for f in copied_videos)
    print(f"   - 总大小: {total_size / (1024*1024):.1f} MB")
    
    return True

if __name__ == "__main__":
    print("📁 创建100个视频的样本数据集")
    print("=" * 50)
    
    success = create_sample_dataset()
    
    if success:
        print("\n🎉 样本数据集创建完成！")
        print("现在可以使用 ./DADA-100-videos 目录进行上传，速度会更快。")
    else:
        print("\n❌ 样本数据集创建失败！")