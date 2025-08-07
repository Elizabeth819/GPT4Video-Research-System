#!/usr/bin/env python3
"""
提交WiseAD视频推理作业到Azure ML
基于YOLO的自动驾驶视频分析 - 低优先级A100版
"""

import os
import sys
import tempfile
import shutil
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.identity import DefaultAzureCredential
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_wisead_code_package():
    """创建WiseAD代码包"""
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="wisead_lowpri_job_")
    logger.info(f"创建临时代码目录: {temp_dir}")
    
    # 需要的文件列表
    required_files = [
        "wisead_video_inference.py",
        "wisead_config.json"
    ]
    
    # 复制必要文件
    for file in required_files:
        if os.path.exists(file):
            shutil.copy2(file, temp_dir)
            logger.info(f"复制文件: {file}")
        else:
            logger.warning(f"文件不存在: {file}")
    
    return temp_dir

def submit_wisead_job():
    """提交WiseAD推理作业"""
    
    logger.info("🚀 提交WiseAD视频推理作业到Azure ML (低优先级A100集群)")
    
    temp_dir = None
    try:
        # 创建代码包
        temp_dir = create_wisead_code_package()
        
        # Azure ML客户端
        credential = DefaultAzureCredential()
        
        # 使用WiseAD工作区
        ml_client = MLClient(
            credential=credential,
            subscription_id="0d3f39ba-7349-4bd7-8122-649ff18f0a4a",
            resource_group_name="wisead-rg",
            workspace_name="wisead-ml-workspace"
        )
        
        logger.info("✅ Azure ML客户端连接成功")
        
        # 使用Azure官方环境
        official_env = "azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10"
        
        # 创建作业
        logger.info("🎯 创建WiseAD视频推理作业 (低优先级A100)...")
        
        job = command(
            display_name="WiseAD_Video_Inference_A100_LowPri_YOLOv8",
            description="基于YOLOv8的WiseAD自动驾驶视频分析系统 - 低优先级A100 GPU版",
            code=temp_dir,
            command="python wisead_video_inference.py --config wisead_config.json",
            environment=official_env,
            compute="wisead-a100-lowpri",
            environment_variables={
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:1024",
                "CUDA_VISIBLE_DEVICES": "0",
                "CUDA_LAUNCH_BLOCKING": "1",
                "TORCH_CUDA_ARCH_LIST": "8.0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=wiseadmlstorage55c2e74d3;AccountKey=26Y8W75xN4RLuTodwDXt6Lz8yPKCRF/+kfOiVaQzD6w+Lhz+KheSI5AwsnEp0F436D016m+nSDXt+AStRZznaQ==;BlobEndpoint=https://wiseadmlstorage55c2e74d3.blob.core.windows.net/;FileEndpoint=https://wiseadmlstorage55c2e74d3.file.core.windows.net/;QueueEndpoint=https://wiseadmlstorage55c2e74d3.queue.core.windows.net/;TableEndpoint=https://wiseadmlstorage55c2e74d3.table.core.windows.net/"
            },
            experiment_name="wisead-a100-lowpri-inference",
            tags={
                "model": "YOLOv8s",
                "task": "autonomous-driving-video-analysis", 
                "framework": "WiseAD",
                "environment": "AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10",
                "compute": "wisead-a100-lowpri",
                "gpu": "A100-LowPriority",
                "optimization": "batch-processing",
                "version": "2.1",
                "cost_optimized": "true"
            }
        )
        
        # 提交作业
        logger.info("📤 提交作业到Azure ML 低优先级A100集群...")
        returned_job = ml_client.create_or_update(job)
        
        logger.info("✅ WiseAD 低优先级A100作业提交成功!")
        logger.info(f"🔗 作业URL: {returned_job.services['Studio'].endpoint}")
        logger.info(f"🆔 作业ID: {returned_job.name}")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 WiseAD视频推理作业已启动! (低优先级A100版)")
        logger.info("📋 作业详情:")
        logger.info(f"   - 模型: YOLOv8s (优化版，平衡速度和精度)")
        logger.info(f"   - 硬件: A100 GPU (80GB显存) - 低优先级")
        logger.info(f"   - 环境: Azure ML PyTorch 1.13")
        logger.info(f"   - 优化: 批处理推理，CUDA优化")
        logger.info(f"   - 成本: 低优先级定价 (约60-80%折扣)")
        logger.info(f"   - 任务: 自动驾驶视频安全分析")
        logger.info(f"   - 分析内容: 车辆检测、行人检测、交通安全评估")
        logger.info(f"   - 性能提升: 批量处理，高频分析，GPU加速")
        logger.info("⏳ 作业将自动安装依赖、下载模型并开始推理...")
        logger.info("💰 低优先级A100 GPU提供成本优化的高性能计算!")
        logger.info("⚠️  注意: 低优先级作业可能会被抢占，但成本更低")
        
        return returned_job.name
        
    except Exception as e:
        logger.error(f"❌ 作业提交失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None
    
    finally:
        # 清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"清理临时目录: {temp_dir}")

if __name__ == "__main__":
    job_id = submit_wisead_job()
    if job_id:
        print(f"\n🎊 成功! WiseAD 低优先级A100作业ID: {job_id}")
        print("请在Azure ML Studio中监控作业进度")
        print("🔗 Azure ML Studio: https://ml.azure.com")
        print(f"💰 低优先级A100 GPU为您提供成本优化的强劲性能!")
        print(f"⚠️  提示: 作业可能会被高优先级作业抢占，但成本节省60-80%")
    else:
        print("\n❌ WiseAD作业提交失败")
        sys.exit(1) 