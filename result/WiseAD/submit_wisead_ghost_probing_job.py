#!/usr/bin/env python3
"""
提交WiseAD A100鬼探头检测作业
使用WiseAD YOLO模型对100个DADA视频进行本地GPU推理
无需外部API，完全基于本地A100 GPU计算
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

def create_wisead_ghost_probing_package():
    """创建WiseAD鬼探头检测代码包"""
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="wisead_ghost_probing_")
    logger.info(f"创建临时代码目录: {temp_dir}")
    
    # 需要的文件列表
    required_files = [
        "azure_ml_wisead_ghost_probing.py",
        "wisead_ghost_probing_config.json"
    ]
    
    # 复制必要文件
    for file in required_files:
        if os.path.exists(file):
            shutil.copy2(file, temp_dir)
            logger.info(f"复制文件: {file}")
        else:
            logger.warning(f"文件不存在: {file}")
    
    return temp_dir

def submit_wisead_ghost_probing_job():
    """提交WiseAD鬼探头检测作业到Azure ML"""
    
    logger.info("🚀 提交WiseAD A100 鬼探头检测作业")
    logger.info("🤖 使用WiseAD YOLO模型进行本地GPU推理")
    
    temp_dir = None
    try:
        # 创建代码包
        temp_dir = create_wisead_ghost_probing_package()
        
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
        logger.info("🎯 创建WiseAD鬼探头检测作业 (低优先级A100)...")
        
        job = command(
            display_name="WiseAD_Ghost_Probing_A100_LowPri",
            description="WiseAD A100 鬼探头检测 - 使用WiseAD YOLO模型对100个DADA视频进行本地推理",
            code=temp_dir,
            command="python azure_ml_wisead_ghost_probing.py --config wisead_ghost_probing_config.json",
            environment=official_env,
            compute="wisead-a100-lowpri",
            environment_variables={
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:1024",
                "CUDA_VISIBLE_DEVICES": "0",
                "CUDA_LAUNCH_BLOCKING": "1",
                "TORCH_CUDA_ARCH_LIST": "8.0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=drivelmmstorage2e932dad7;AccountKey=MniZTrPWLKwVg6XpJKpu+4Rv5fuvd0x+xq2smYW+yZn1IGVpf5OcMuGfLBmSuKyOWhAjOLGnbNIq+AStpd49zQ==;EndpointSuffix=core.windows.net"
            },
            experiment_name="wisead-ghost-probing-detection",
            tags={
                "task": "ghost-probing-detection",
                "model": "WiseAD-YOLO-v8", 
                "framework": "Local-GPU-Inference",
                "compute": "wisead-a100-lowpri",
                "gpu": "A100-LowPriority",
                "videos": "100-DADA-videos",
                "inference_type": "local_model",
                "no_external_api": "true",
                "yolo_version": "v8s",
                "confidence_threshold": "0.5",
                "version": "1.0"
            }
        )
        
        # 提交作业
        logger.info("📤 提交作业到Azure ML 低优先级A100集群...")
        returned_job = ml_client.create_or_update(job)
        
        logger.info("✅ WiseAD鬼探头检测作业提交成功!")
        logger.info(f"🔗 作业URL: {returned_job.services['Studio'].endpoint}")
        logger.info(f"🆔 作业ID: {returned_job.name}")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 WiseAD A100 鬼探头检测作业已启动!")
        logger.info("📋 作业详情:")
        logger.info(f"   - 任务: WiseAD鬼探头检测")
        logger.info(f"   - 模型: WiseAD YOLO v8s (本地推理)")
        logger.info(f"   - 硬件: A100 GPU (80GB显存) - 低优先级")
        logger.info(f"   - 视频: 100个DADA视频 (images_1_001 - images_5_XXX)")
        logger.info(f"   - 推理方式: 本地GPU推理，无外部API")
        logger.info(f"   - 置信度阈值: 0.5")
        logger.info(f"   - 帧分析间隔: 每3帧")
        logger.info("⏳ 作业将自动:")
        logger.info("   1. 从Azure Storage下载100个DADA视频")
        logger.info("   2. 使用WiseAD YOLO模型进行目标检测")
        logger.info("   3. 基于检测结果分析鬼探头行为")
        logger.info("   4. 生成详细的鬼探头检测报告")
        logger.info("🤖 完全本地推理，充分利用A100 GPU性能!")
        logger.info("⚠️  注意: 低优先级作业可能会被抢占，但成本节省60-80%")
        
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
    job_id = submit_wisead_ghost_probing_job()
    if job_id:
        print(f"\n🎊 成功! WiseAD鬼探头检测作业ID: {job_id}")
        print("请在Azure ML Studio中监控作业进度")
        print("🔗 Azure ML Studio: https://ml.azure.com")
        print("\n🤖 WiseAD系统特点:")
        print(f"   - 本地A100 GPU推理，无需外部API")
        print(f"   - YOLOv8s模型，专业目标检测")
        print(f"   - 实时鬼探头行为分析算法")
        print(f"   - 突然出现、危险距离、意外运动检测")
        print(f"   - 完全自主的智能驾驶场景分析")
    else:
        print("\n❌ WiseAD鬼探头检测作业提交失败")
        sys.exit(1) 