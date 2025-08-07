#!/usr/bin/env python3
"""
提交Azure ML A100鬼探头分析作业
使用GPT-4.1 Balanced Prompt对100个DADA视频进行鬼探头标注
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

def create_ghost_probing_code_package():
    """创建鬼探头分析代码包"""
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="azure_ghost_probing_")
    logger.info(f"创建临时代码目录: {temp_dir}")
    
    # 需要的文件列表
    required_files = [
        "azure_ml_ghost_probing_gpt41_balanced.py",
        "azure_ghost_probing_config.json"
    ]
    
    # 复制必要文件
    for file in required_files:
        if os.path.exists(file):
            shutil.copy2(file, temp_dir)
            logger.info(f"复制文件: {file}")
        else:
            logger.warning(f"文件不存在: {file}")
    
    return temp_dir

def submit_ghost_probing_job():
    """提交鬼探头分析作业到Azure ML"""
    
    logger.info("🚀 提交Azure ML A100 鬼探头分析作业")
    logger.info("👻 使用GPT-4.1 Balanced Prompt进行公平对比")
    
    temp_dir = None
    try:
        # 创建代码包
        temp_dir = create_ghost_probing_code_package()
        
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
        logger.info("🎯 创建鬼探头分析作业 (低优先级A100)...")
        
        job = command(
            display_name="Ghost_Probing_GPT41_Balanced_A100_LowPri",
            description="Azure ML A100 鬼探头分析 - 使用GPT-4.1 Balanced Prompt对100个DADA视频进行标注",
            code=temp_dir,
            command="python azure_ml_ghost_probing_gpt41_balanced.py --config azure_ghost_probing_config.json",
            environment=official_env,
            compute="wisead-a100-lowpri",
            environment_variables={
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:1024",
                "CUDA_VISIBLE_DEVICES": "0",
                "CUDA_LAUNCH_BLOCKING": "1",
                "TORCH_CUDA_ARCH_LIST": "8.0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=drivelmmstorage2e932dad7;AccountKey=MniZTrPWLKwVg6XpJKpu+4Rv5fuvd0x+xq2smYW+yZn1IGVpf5OcMuGfLBmSuKyOWhAjOLGnbNIq+AStpd49zQ==;EndpointSuffix=core.windows.net",
                "AZURE_OPENAI_KEY": os.getenv('AZURE_OPENAI_KEY', ''),
                "AZURE_OPENAI_ENDPOINT": os.getenv('AZURE_OPENAI_ENDPOINT', ''),
                "AZURE_OPENAI_DEPLOYMENT": os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o'),
                "AZURE_OPENAI_API_VERSION": os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
            },
            experiment_name="ghost-probing-gpt41-balanced",
            tags={
                "task": "ghost-probing-analysis",
                "model": "GPT-4.1-Balanced", 
                "framework": "Azure-OpenAI",
                "compute": "wisead-a100-lowpri",
                "gpu": "A100-LowPriority",
                "videos": "100-DADA-videos",
                "comparison": "GPT41-Balanced-Baseline",
                "baseline_f1": "0.712",
                "baseline_recall": "0.963",
                "baseline_precision": "0.565",
                "prompt_version": "identical_to_baseline",
                "version": "1.0"
            }
        )
        
        # 提交作业
        logger.info("📤 提交作业到Azure ML 低优先级A100集群...")
        returned_job = ml_client.create_or_update(job)
        
        logger.info("✅ 鬼探头分析作业提交成功!")
        logger.info(f"🔗 作业URL: {returned_job.services['Studio'].endpoint}")
        logger.info(f"🆔 作业ID: {returned_job.name}")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 Azure ML A100 鬼探头分析作业已启动!")
        logger.info("📋 作业详情:")
        logger.info(f"   - 任务: 鬼探头标注分析")
        logger.info(f"   - 模型: GPT-4.1 Balanced (完全相同prompt)")
        logger.info(f"   - 硬件: A100 GPU (80GB显存) - 低优先级")
        logger.info(f"   - 视频: 100个DADA视频 (images_1_001 - images_5_XXX)")
        logger.info(f"   - 对比基准: GPT-4.1 Balanced (F1=0.712)")
        logger.info(f"   - 输出格式: 与GPT-4.1完全一致的JSON格式")
        logger.info(f"   - 评估指标: 准确率、精确度、召回率、F1分数")
        logger.info("⏳ 作业将自动:")
        logger.info("   1. 从Azure Storage下载100个DADA视频")
        logger.info("   2. 使用GPT-4.1 Balanced Prompt进行分析")
        logger.info("   3. 生成与baseline完全一致的JSON结果")
        logger.info("   4. 计算对比性能指标")
        logger.info("💰 低优先级A100 GPU提供成本优化的强劲性能!")
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
    job_id = submit_ghost_probing_job()
    if job_id:
        print(f"\n🎊 成功! 鬼探头分析作业ID: {job_id}")
        print("请在Azure ML Studio中监控作业进度")
        print("🔗 Azure ML Studio: https://ml.azure.com")
        print("\n📊 预期结果:")
        print(f"   - 处理100个DADA视频的鬼探头标注")
        print(f"   - 与GPT-4.1 Balanced (F1=0.712) 进行公平对比")
        print(f"   - 生成准确率、精确度、召回率等详细指标")
        print(f"   - 输出格式完全一致，便于直接比较分析")
    else:
        print("\n❌ 鬼探头分析作业提交失败")
        sys.exit(1) 