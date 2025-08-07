#!/usr/bin/env python3
"""
WiseAD专用100个视频上传脚本
上传images_1_001到images_5_XXX系列的100个真实DADA视频到Azure Storage
专为WiseAD推理系统设计
"""

import os
import sys
import logging
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_env_file():
    """加载.env文件中的环境变量"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key == 'AZURE_STORAGE_CONNECTION_STRING':
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
                        break

def check_azure_connection():
    """检查Azure Storage连接字符串"""
    connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        logger.error("❌ 未找到AZURE_STORAGE_CONNECTION_STRING环境变量")
        logger.info("💡 请设置连接字符串:")
        logger.info("   export AZURE_STORAGE_CONNECTION_STRING='your_connection_string'")
        return None
    
    logger.info("✅ 找到Azure Storage连接字符串")
    return connection_string

def get_target_100_videos():
    """获取目标100个视频文件"""
    logger.info("🔍 搜索目标100个视频文件...")
    
    try:
        # 从文件读取视频列表
        if os.path.exists('target_100_videos.txt'):
            with open('target_100_videos.txt', 'r') as f:
                video_paths = [line.strip() for line in f.readlines()]
            
            # 验证文件存在
            valid_videos = []
            for video_path in video_paths:
                if os.path.exists(video_path):
                    valid_videos.append(video_path)
                else:
                    logger.warning(f"⚠️ 文件不存在: {video_path}")
            
            logger.info(f"📹 找到 {len(valid_videos)} 个有效视频文件")
            return valid_videos[:100]  # 确保只取100个
        else:
            logger.error("❌ 未找到target_100_videos.txt文件")
            return []
            
    except Exception as e:
        logger.error(f"❌ 获取视频列表失败: {e}")
        return []

def upload_single_video_with_progress(connection_string, container_name, video_path, video_index, total_videos):
    """上传单个视频文件（带进度显示）"""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.error("❌ 缺少azure-storage-blob依赖")
        return False, "依赖缺失", 0
    
    video_name = os.path.basename(video_path)
    blob_name = video_name
    
    try:
        start_time = time.time()
        
        # 创建blob客户端
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        # 检查文件是否已存在
        try:
            if blob_client.exists():
                logger.info(f"   [{video_index:3d}/{total_videos}] ⚠️ 文件已存在，跳过: {video_name}")
                return True, "已存在", 0
        except Exception:
            pass
        
        # 获取文件大小
        file_size = os.path.getsize(video_path) / 1024 / 1024  # MB
        logger.info(f"   [{video_index:3d}/{total_videos}] 📤 开始上传: {video_name} ({file_size:.1f}MB)")
        
        # 上传文件
        with open(video_path, "rb") as data:
            blob_client.upload_blob(
                data, 
                overwrite=True,
                max_concurrency=3,
                timeout=1800  # 30分钟超时
            )
        
        upload_time = time.time() - start_time
        logger.info(f"   [{video_index:3d}/{total_videos}] ✅ 上传成功: {video_name} ({upload_time:.1f}s)")
        return True, f"上传成功", upload_time
        
    except Exception as e:
        upload_time = time.time() - start_time
        logger.error(f"   [{video_index:3d}/{total_videos}] ❌ 上传失败 {video_name}: {e}")
        return False, str(e), upload_time

def save_upload_progress(results, success_count, total_videos):
    """保存上传进度"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_file = f"wisead_upload_progress_{timestamp}.json"
    
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "total_videos": total_videos,
        "success_count": success_count,
        "failed_count": total_videos - success_count,
        "progress_percentage": (success_count / total_videos) * 100,
        "detailed_results": results
    }
    
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📋 上传进度已保存: {progress_file}")

def main():
    """主函数"""
    logger.info("🚀 WiseAD专用100个视频上传到Azure Storage")
    logger.info("📍 上传范围: images_1_001 到 images_5_XXX (100个视频)")
    logger.info("🎯 目标Storage Account: wisead系列")
    logger.info("=" * 70)
    
    # 加载环境变量
    load_env_file()
    
    try:
        # 1. 检查Azure连接
        connection_string = check_azure_connection()
        if not connection_string:
            return 1
        
        # 2. 检查依赖
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            logger.error("❌ 缺少azure-storage-blob依赖")
            logger.info("💡 请安装: pip install azure-storage-blob")
            return 1
        
        # 3. 创建Azure客户端
        logger.info("🔗 连接Azure Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # 4. 设置容器名称（WiseAD专用）
        container_name = 'wisead-videos'  # WiseAD专用容器
        logger.info(f"📦 使用WiseAD专用容器: {container_name}")
        
        # 5. 确保容器存在
        try:
            container_client = blob_service_client.get_container_client(container_name)
            if not container_client.exists():
                logger.info(f"📦 创建WiseAD专用容器: {container_name}")
                container_client.create_container()
            else:
                logger.info(f"✅ WiseAD容器已存在: {container_name}")
        except Exception as e:
            logger.error(f"❌ 容器操作失败: {e}")
            return 1
        
        # 6. 获取要上传的视频
        videos_to_upload = get_target_100_videos()
        if not videos_to_upload:
            logger.error("❌ 没有找到视频文件")
            return 1
        
        if len(videos_to_upload) != 100:
            logger.warning(f"⚠️ 找到 {len(videos_to_upload)} 个视频，预期100个")
        
        # 7. 开始批量上传
        logger.info(f"\n📤 开始批量上传 {len(videos_to_upload)} 个视频...")
        logger.info(f"⚡ 使用并行上传（最大并发数: 3）")
        
        start_time = time.time()
        results = []
        success_count = 0
        
        # 使用ThreadPoolExecutor进行并行上传
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交所有上传任务
            future_to_video = {
                executor.submit(
                    upload_single_video_with_progress,
                    connection_string,
                    container_name,
                    video_path,
                    i + 1,
                    len(videos_to_upload)
                ): (video_path, i + 1) for i, video_path in enumerate(videos_to_upload)
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_video):
                video_path, video_index = future_to_video[future]
                video_name = os.path.basename(video_path)
                
                try:
                    success, message, upload_time = future.result()
                    if success:
                        success_count += 1
                    
                    results.append({
                        "video_name": video_name,
                        "video_index": video_index,
                        "success": success,
                        "message": message,
                        "upload_time": upload_time
                    })
                    
                except Exception as e:
                    logger.error(f"❌ 任务处理失败 {video_name}: {e}")
                    results.append({
                        "video_name": video_name,
                        "video_index": video_index,
                        "success": False,
                        "message": str(e),
                        "upload_time": 0
                    })
        
        # 8. 显示最终结果
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("🎉 WiseAD视频上传完成！")
        logger.info(f"📊 上传统计:")
        logger.info(f"   成功: {success_count}/{len(videos_to_upload)}")
        logger.info(f"   失败: {len(videos_to_upload) - success_count}")
        logger.info(f"   成功率: {(success_count / len(videos_to_upload)) * 100:.1f}%")
        logger.info(f"   总耗时: {total_time:.1f}秒")
        
        # 9. 保存进度
        save_upload_progress(results, success_count, len(videos_to_upload))
        
        # 10. 更新WiseAD配置
        logger.info("\n🔧 更新WiseAD配置...")
        wisead_config = {
            "azure_storage_container": container_name,
            "uploaded_videos_count": success_count,
            "upload_timestamp": datetime.now().isoformat(),
            "batch_size": 4,
            "confidence_threshold": 0.5,
            "model_type": "yolov8",
            "max_videos": success_count  # 使用实际上传的视频数量
        }
        
        # 更新wisead_config.json
        config_file = "wisead_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                existing_config = json.load(f)
            
            # 合并配置
            existing_config.update(wisead_config)
            existing_config["parameters"].update({
                "max_videos": success_count
            })
            
            with open(config_file, 'w') as f:
                json.dump(existing_config, f, indent=2)
            
            logger.info(f"✅ WiseAD配置已更新: {config_file}")
        
        if success_count == len(videos_to_upload):
            logger.info("🎯 所有视频上传成功！WiseAD系统已准备就绪")
            return 0
        else:
            logger.warning(f"⚠️ 部分视频上传失败，成功率: {(success_count / len(videos_to_upload)) * 100:.1f}%")
            return 1
        
    except Exception as e:
        logger.error(f"❌ 上传过程异常: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 