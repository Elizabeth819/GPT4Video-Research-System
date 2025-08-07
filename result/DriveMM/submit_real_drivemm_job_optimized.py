#!/usr/bin/env python3
"""
优化版本：提交真实DriveMM推理作业到Azure ML
只上传必要的文件，避免上传大量不必要的数据
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

def create_minimal_code_package():
    """创建最小化的代码包，只包含必要文件"""
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="drivemm_job_")
    logger.info(f"创建临时代码目录: {temp_dir}")
    
    # 需要的文件列表
    required_files = [
        "azure_drivemm_real_inference.py",
        "config.json",
        "requirements.txt"
    ]
    
    # 复制必要文件
    for file in required_files:
        if os.path.exists(file):
            shutil.copy2(file, temp_dir)
            logger.info(f"复制文件: {file}")
        else:
            logger.warning(f"文件不存在: {file}")
    
    return temp_dir

def submit_real_drivemm_job():
    """提交真实DriveMM推理作业"""
    
    logger.info("🚀 提交优化版真实DriveMM推理作业到Azure ML")
    
    temp_dir = None
    try:
        # 创建最小化代码包
        temp_dir = create_minimal_code_package()
        
        # Azure ML客户端
        credential = DefaultAzureCredential()
        ml_client = MLClient.from_config(credential=credential)
        
        logger.info("✅ Azure ML客户端连接成功")
        
        # 直接使用Azure官方环境
        official_env = "azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10"
        
        # 创建作业
        logger.info("🎯 创建真实DriveMM推理作业...")
        
        job = command(
            display_name="DriveMM_Real_Inference_Ghost_Probing_Optimized",
            description="使用真实DriveMM模型(17GB)在GPU环境中进行鬼探头检测 - 优化版本",
            code=temp_dir,  # 使用临时目录而不是当前目录
            command="pip install transformers>=4.35.0 huggingface_hub>=0.20.0 accelerate>=0.24.0 azure-storage-blob>=12.19.0 opencv-python>=4.8.0 && python azure_drivemm_real_inference.py",
            environment=official_env,
            compute="drivemm-a100-cluster",
            environment_variables={
                "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=drivelmmstorage2e932dad7;AccountKey=MniZTrPWLKwVg6XpJKpu+4Rv5fuvd0x+xq2smYW+yZn1IGVpf5OcMuGfLBmSuKyOWhAjOLGnbNIq+AStpd49zQ==;EndpointSuffix=core.windows.net",
                "HF_HOME": "/tmp/huggingface_cache",
                "TRANSFORMERS_CACHE": "/tmp/transformers_cache",
                "TORCH_HOME": "/tmp/torch_cache"
            },
            experiment_name="drivemm-real-inference",
            tags={
                "model": "DriveMM-Real-17GB",
                "task": "ghost-probing-detection", 
                "comparison": "vs-GPT41-balanced",
                "environment": "AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10",
                "model_source": "huggingface.co/DriveMM/DriveMM",
                "version": "optimized"
            }
        )
        
        # 提交作业
        logger.info("📤 提交作业到Azure ML...")
        returned_job = ml_client.create_or_update(job)
        
        logger.info("✅ 真实DriveMM作业提交成功!")
        logger.info(f"🔗 作业URL: {returned_job.services['Studio'].endpoint}")
        logger.info(f"🆔 作业ID: {returned_job.name}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 优化版真实DriveMM推理作业已启动!")
        logger.info("📋 作业详情:")
        logger.info(f"   - 模型: DriveMM/DriveMM (真实17GB模型)")
        logger.info(f"   - 环境: Azure ML GPU")
        logger.info(f"   - 数据源: drivelmmstorage2e932dad7")
        logger.info(f"   - 任务: 99个视频鬼探头检测")
        logger.info(f"   - 对比基准: GPT-4.1 Balanced F1=0.712")
        logger.info("⏳ 作业将自动下载DriveMM模型并开始推理...")
        
        return returned_job.name
        
    except Exception as e:
        logger.error(f"❌ 作业提交失败: {e}")
        return None
    
    finally:
        # 清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"清理临时目录: {temp_dir}")

if __name__ == "__main__":
    job_id = submit_real_drivemm_job()
    if job_id:
        print(f"\n🎊 成功! 作业ID: {job_id}")
        print("请在Azure ML Studio中监控作业进度")
    else:
        print("\n❌ 作业提交失败")
        sys.exit(1) 