#!/usr/bin/env python3
"""
批量视频DriveMM分析脚本
"""

import os
import sys
import json
import glob
from pathlib import Path
import argparse
from datetime import datetime
import logging
from single_video_analysis import extract_video_frames, analyze_with_drivemm_demo

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def batch_analyze_videos(video_dir, output_dir, limit=5):
    """批量分析视频"""
    logger.info(f"🎬 批量分析开始")
    logger.info(f"📁 视频目录: {video_dir}")
    logger.info(f"📤 输出目录: {output_dir}")
    logger.info(f"🔢 处理限制: {limit} 个视频")
    
    # 获取视频文件列表
    video_files = glob.glob(os.path.join(video_dir, "images_*.avi"))
    video_files.sort()
    
    if limit:
        video_files = video_files[:limit]
    
    logger.info(f"📊 找到 {len(video_files)} 个视频文件")
    
    results = []
    ghost_detections = 0
    start_time = datetime.now()
    
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"\n🎯 处理视频 {i}/{len(video_files)}: {os.path.basename(video_path)}")
        
        try:
            # 提取帧
            frames, frame_info = extract_video_frames(video_path, num_frames=3)  # 减少帧数提高速度
            
            # 分析
            result = analyze_with_drivemm_demo(video_path, frames, frame_info)
            results.append(result)
            
            # 统计鬼探头检测
            if result["analysis_results"]["ghost_probing"]["detected"]:
                ghost_detections += 1
                logger.info(f"   🚨 检测到鬼探头! 总计: {ghost_detections}")
            else:
                logger.info(f"   ✅ 正常场景")
            
            # 保存单个结果
            video_name = os.path.basename(video_path).replace('.avi', '')
            result_file = os.path.join(output_dir, f"drivemm_analysis_{video_name}.json")
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"❌ 处理 {video_path} 失败: {e}")
            continue
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # 生成批量分析报告
    summary = {
        "batch_analysis_summary": {
            "total_videos": len(results),
            "ghost_probing_detected": ghost_detections,
            "detection_rate": ghost_detections / len(results) if results else 0,
            "processing_time_seconds": processing_time,
            "average_time_per_video": processing_time / len(results) if results else 0,
            "method": "DriveMM_Demo_Batch_Analysis",
            "timestamp": datetime.now().isoformat()
        },
        "detailed_results": results
    }
    
    # 保存批量报告
    summary_file = os.path.join(output_dir, "drivemm_batch_analysis_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 显示结果
    logger.info("\n🎉 批量分析完成!")
    logger.info("=" * 50)
    logger.info(f"📊 处理统计:")
    logger.info(f"   总视频数: {len(results)}")
    logger.info(f"   鬼探头检测: {ghost_detections} 个")
    logger.info(f"   检测率: {ghost_detections / len(results):.1%}" if results else "N/A")
    logger.info(f"   总处理时间: {processing_time:.2f} 秒")
    logger.info(f"   平均处理时间: {processing_time / len(results):.2f} 秒/视频" if results else "N/A")
    logger.info(f"📁 批量报告: {summary_file}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='DriveMM批量视频分析')
    parser.add_argument('--video_dir', 
                       default='/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos',
                       help='视频目录路径')
    parser.add_argument('--output_dir', 
                       default='/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DriveMM/batch_analysis_results', 
                       help='输出目录')
    parser.add_argument('--limit', type=int, default=10, help='处理视频数量限制')
    
    args = parser.parse_args()
    
    # 检查视频目录
    if not os.path.exists(args.video_dir):
        logger.error(f"❌ 视频目录不存在: {args.video_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("🚀 DriveMM 批量视频分析")
    logger.info("=" * 50)
    
    # 执行批量分析
    summary = batch_analyze_videos(args.video_dir, args.output_dir, args.limit)
    
    logger.info(f"\n✅ 批量分析成功完成!")

if __name__ == "__main__":
    main()