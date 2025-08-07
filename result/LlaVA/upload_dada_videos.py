#!/usr/bin/env python3
"""
Upload DADA-100 Videos to Azure ML Datastore
上传DADA-100视频到Azure ML数据存储
文件路径: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/upload_dada_videos.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Data
    from azure.ai.ml.constants import AssetTypes
    from azure.identity import DefaultAzureCredential
except ImportError:
    print("❌ Azure ML SDK未安装，请运行: pip install azure-ai-ml")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DADAVideoUploader:
    """DADA视频上传器"""
    
    def __init__(self, 
                 subscription_id: str = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a",
                 resource_group: str = "llava-resourcegroup", 
                 workspace_name: str = "llava-workspace"):
        """
        初始化Azure ML客户端
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        
        try:
            # 初始化ML客户端
            credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
            
            logger.info(f"✅ Azure ML客户端初始化成功")
            logger.info(f"📋 订阅: {subscription_id}")
            logger.info(f"📋 资源组: {resource_group}")
            logger.info(f"📋 工作区: {workspace_name}")
            
        except Exception as e:
            logger.error(f"❌ Azure ML客户端初始化失败: {e}")
            raise
    
    def upload_dada_videos(self, local_path: str, data_name: str = "DADA-100-videos"):
        """
        上传DADA视频到Azure ML数据存储
        
        Args:
            local_path: 本地DADA视频目录路径
            data_name: Azure ML中的数据资产名称
        """
        try:
            local_path = Path(local_path)
            
            # 检查本地路径
            if not local_path.exists():
                raise FileNotFoundError(f"本地路径不存在: {local_path}")
            
            # 统计视频文件
            video_files = list(local_path.glob("*.avi"))
            logger.info(f"📁 本地路径: {local_path}")
            logger.info(f"🎬 发现视频文件: {len(video_files)}个")
            
            if len(video_files) == 0:
                raise ValueError("未找到.avi视频文件")
            
            # 创建数据资产
            logger.info(f"📤 开始上传到Azure ML数据存储...")
            
            data_asset = Data(
                path=str(local_path),
                type=AssetTypes.URI_FOLDER,
                description=f"DADA-100 Ghost Probing Video Dataset ({len(video_files)} videos)",
                name=data_name,
                version=datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            
            # 上传数据
            uploaded_data = self.ml_client.data.create_or_update(data_asset)
            
            logger.info(f"✅ 数据上传成功!")
            logger.info(f"📊 数据资产名称: {uploaded_data.name}")
            logger.info(f"🔢 版本: {uploaded_data.version}")
            logger.info(f"📁 Azure路径: {uploaded_data.path}")
            
            # 输出可用于作业的路径
            azure_path = f"azureml:{data_name}:{uploaded_data.version}"
            logger.info(f"🔗 作业中使用的路径: {azure_path}")
            
            return uploaded_data
            
        except Exception as e:
            logger.error(f"❌ 数据上传失败: {e}")
            raise
    
    def list_data_assets(self):
        """列出数据资产"""
        try:
            logger.info("📋 当前数据资产:")
            logger.info("=" * 60)
            
            data_assets = list(self.ml_client.data.list())
            
            if not data_assets:
                logger.info("📂 暂无数据资产")
                return
            
            for asset in data_assets:
                logger.info(f"📊 名称: {asset.name}")
                logger.info(f"🔢 版本: {asset.version}")
                logger.info(f"📝 描述: {asset.description}")
                logger.info(f"📁 路径: {asset.path}")
                logger.info("-" * 40)
                
        except Exception as e:
            logger.error(f"❌ 列出数据资产失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DADA视频上传工具')
    parser.add_argument('--local-path', type=str, 
                       default='../../result/DADA-100-videos',
                       help='本地DADA视频目录路径')
    parser.add_argument('--data-name', type=str, 
                       default='DADA-100-videos',
                       help='Azure ML中的数据资产名称')
    parser.add_argument('--list-only', action='store_true',
                       help='只列出现有数据资产')
    
    args = parser.parse_args()
    
    try:
        uploader = DADAVideoUploader()
        
        if args.list_only:
            uploader.list_data_assets()
        else:
            # 上传数据
            uploaded_data = uploader.upload_dada_videos(args.local_path, args.data_name)
            
            print("\n" + "="*60)
            print("🎉 数据上传完成!")
            print("="*60)
            print(f"📊 数据资产: {uploaded_data.name}:{uploaded_data.version}")
            print(f"🔗 作业路径: azureml:{uploaded_data.name}:{uploaded_data.version}")
            print("="*60)
            
            print("\n💡 现在可以重新提交LLaVA作业：")
            print("python submit_azure_llava_job.py --action submit --limit 100 --no-dry-run")
            
    except Exception as e:
        logger.error(f"❌ 操作失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()