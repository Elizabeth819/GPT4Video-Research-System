#!/usr/bin/env python3
"""
提交真实DriveMM推理作业到Azure ML
"""

import os
import sys
from azure.ai.ml import MLClient
from azure.ai.ml.entities import CommandJob, Environment
from azure.identity import DefaultAzureCredential
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def submit_drivemm_job():
    """提交DriveMM推理作业到Azure ML"""
    
    try:
        # Azure ML配置 - 请根据您的实际配置修改
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID", "your-subscription-id")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP", "your-resource-group")
        workspace_name = os.getenv("AZURE_ML_WORKSPACE", "your-workspace-name")
        
        logger.info("🔗 连接到Azure ML...")
        
        # 创建ML客户端
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        
        logger.info("✅ Azure ML连接成功")
        
        # 创建作业
        logger.info("📋 创建DriveMM推理作业...")
        
        job = CommandJob(
            display_name="real-drivemm-ghost-probing-inference",
            description="真实DriveMM模型鬼探头检测推理 - 与GPT-4.1公平对比",
            tags={
                "model": "DriveMM-8.45B",
                "task": "ghost_probing_detection", 
                "comparison": "GPT-4.1_vs_DriveMM",
                "data_source": "drivelmmstorage2e932dad7"
            },
            
            # 计算配置
            compute="gpu-cluster",  # 请根据您的GPU计算集群名称修改
            
            # 环境配置
            environment="azureml://registries/azureml/environments/pytorch-1.13-ubuntu20.04-py38-cuda11.6-gpu/versions/latest",
            
            # 代码
            code="./",
            
            # 命令
            command="""
            pip install --upgrade pip &&
            pip install transformers>=4.25.0 torch torchvision torchaudio &&
            pip install azure-storage-blob azure-identity &&
            pip install opencv-python pillow numpy &&
            pip install huggingface_hub accelerate &&
            git clone https://github.com/zhijian11/DriveMM.git &&
            cd DriveMM && pip install -e . && cd .. &&
            python azure_drivemm_real_inference.py
            """,
            
            # 环境变量
            environment_variables={
                "AZURE_STORAGE_ACCOUNT": "drivelmmstorage2e932dad7",
                "HF_HOME": "/tmp/huggingface",
                "TRANSFORMERS_CACHE": "/tmp/transformers",
                "CUDA_VISIBLE_DEVICES": "0"
            },
            
            # 资源配置
            instance_type="Standard_NC6s_v3",  # V100 GPU
            timeout=14400,  # 4小时
        )
        
        # 提交作业
        logger.info("🚀 提交作业到Azure ML...")
        submitted_job = ml_client.jobs.create_or_update(job)
        
        logger.info(f"✅ 作业提交成功!")
        logger.info(f"📊 作业名称: {submitted_job.name}")
        logger.info(f"📊 作业ID: {submitted_job.id}")
        logger.info(f"📊 作业状态: {submitted_job.status}")
        logger.info(f"🔗 作业URL: {submitted_job.studio_url}")
        
        # 可选：等待作业完成
        print("\n是否等待作业完成? (y/n): ", end="")
        wait_for_completion = input().lower() == 'y'
        
        if wait_for_completion:
            logger.info("⏳ 等待作业完成...")
            final_job = ml_client.jobs.stream(submitted_job.name)
            logger.info(f"🎉 作业完成! 状态: {final_job.status}")
        
        return submitted_job
        
    except Exception as e:
        logger.error(f"❌ 作业提交失败: {e}")
        raise

def main():
    """主函数"""
    try:
        # 检查必要文件
        required_files = [
            "azure_drivemm_real_inference.py",
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"❌ 缺少必要文件: {file}")
                return 1
        
        # 提交作业
        job = submit_drivemm_job()
        
        logger.info("\n" + "="*60)
        logger.info("🎯 作业提交完成!")
        logger.info("📋 后续步骤:")
        logger.info("1. 在Azure ML Studio中监控作业进度")
        logger.info("2. 作业完成后下载结果文件")
        logger.info("3. 分析DriveMM vs GPT-4.1的对比结果")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)