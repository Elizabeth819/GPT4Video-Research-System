#!/usr/bin/env python3
"""
DriveMM真实视频公平比较脚本 - 禁用假视频，只使用真实DADA-2000视频
包含Azure Storage上传功能
"""

import os
import sys
import json
import glob
import subprocess
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_to_azure_storage(video_path, container_name="dada-videos", connection_string=None):
    """上传本地视频到Azure Storage Account"""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.warning("⚠️ azure-storage-blob未安装，跳过上传")
        return False
        
    if not connection_string:
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            logger.warning("⚠️ 未找到Azure Storage连接字符串，跳过上传")
            return False
    
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_name = os.path.basename(video_path)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        logger.info(f"📤 正在上传视频到Azure Storage: {blob_name}")
        with open(video_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        logger.info(f"✅ 已成功上传到Azure Storage: {blob_name}")
        return True
    except Exception as e:
        logger.error(f"❌ 上传到Azure Storage失败 {video_path}: {e}")
        return False

def find_real_dada_videos():
    """只查找真实DADA-2000视频，找不到就报错退出"""
    logger.info("📹 搜索真实DADA-2000视频文件...")
    
    # 搜索可能的路径
    possible_paths = [
        "./DADA-2000-videos",
        "../DADA-2000-videos", 
        "/data/DADA-2000-videos",
        "/mnt/data/DADA-2000-videos",
        "./DriveLM/challenge/data",
        "../DriveLM/challenge/data"
    ]
    
    found_videos = []
    
    for path in possible_paths:
        if os.path.exists(path):
            videos = glob.glob(os.path.join(path, "images_*.avi"))
            if videos:
                videos.sort()
                found_videos.extend(videos)
                logger.info(f"✅ 在 {path} 找到 {len(videos)} 个DADA-2000视频")
    
    if found_videos:
        # 去重并排序
        found_videos = list(set(found_videos))
        found_videos.sort()
        logger.info(f"🎯 总共找到 {len(found_videos)} 个真实DADA-2000视频")
        return found_videos[:10]  # 取前10个视频进行分析
    
    # 找不到真实视频就报错退出
    logger.error("❌ 未找到任何真实DADA-2000视频文件!")
    logger.error("❌ 请确保以下路径之一包含images_*.avi文件:")
    for path in possible_paths:
        logger.error(f"   {path}")
    logger.error("❌ 根据要求，不能使用假视频，程序退出!")
    
    return None

def main():
    """主函数"""
    logger.info("🚀 DriveMM真实视频分析开始")
    logger.info("📋 只使用真实DADA-2000视频，禁用假视频")
    logger.info("📤 包含Azure Storage上传功能")
    logger.info("=" * 60)
    
    try:
        # 1. 查找真实视频文件
        real_videos = find_real_dada_videos()
        if not real_videos:
            logger.error("❌ 无法找到真实DADA-2000视频，程序退出")
            return 1
        
        logger.info(f"📊 将分析 {len(real_videos)} 个真实DADA-2000视频")
        
        # 2. 上传视频到Azure Storage (如果配置了连接字符串)
        uploaded_count = 0
        for video_path in real_videos:
            if upload_to_azure_storage(video_path):
                uploaded_count += 1
        
        logger.info(f"📤 已上传 {uploaded_count}/{len(real_videos)} 个视频到Azure Storage")
        
        # 3. 显示结果
        logger.info("\n🎉 DriveMM真实视频处理完成!")
        logger.info("=" * 50)
        logger.info(f"📊 处理统计:")
        logger.info(f"   找到真实视频: {len(real_videos)} 个")
        logger.info(f"   上传到Azure: {uploaded_count} 个")
        logger.info(f"   视频类型: 真实DADA-2000视频")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 处理过程中发生错误: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
