#!/usr/bin/env python3
"""
上传DADA-100视频到Azure ML工作区存储
"""

import os
import sys
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_videos():
    """上传视频到Azure ML"""
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
        from azure.ai.ml.entities import Data
        from azure.ai.ml.constants import AssetTypes
        
        # Azure ML客户端
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id="0d3f39ba-7349-4bd7-8122-649ff18f0a4a",
            resource_group_name="llava-resourcegroup",
            workspace_name="llava-workspace"
        )
        
        # 本地视频文件夹
        video_folder = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos"
        
        if not Path(video_folder).exists():
            logger.error(f"视频文件夹不存在: {video_folder}")
            return False
        
        logger.info(f"正在上传视频文件夹: {video_folder}")
        
        # 创建数据资产
        data_asset = Data(
            path=video_folder,
            type=AssetTypes.URI_FOLDER,
            description="DADA-100 videos for ghost probing detection",
            name="dada-100-videos-fixed"
        )
        
        # 上传数据
        logger.info("开始上传数据到Azure ML...")
        uploaded_data = ml_client.data.create_or_update(data_asset)
        
        logger.info(f"✅ 数据上传成功!")
        logger.info(f"数据资产名称: {uploaded_data.name}")
        logger.info(f"数据资产版本: {uploaded_data.version}")
        logger.info(f"数据路径: {uploaded_data.path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 上传失败: {e}")
        return False

def main():
    success = upload_videos()
    if success:
        logger.info("🎉 视频上传完成!")
    else:
        logger.error("❌ 视频上传失败")
        sys.exit(1)

if __name__ == "__main__":
    main()