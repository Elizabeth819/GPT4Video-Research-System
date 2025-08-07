#!/usr/bin/env python
"""
设置DriveMM在Azure ML上的推理环境
"""

import os
import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, CommandJob, AmlCompute
from azure.identity import DefaultAzureCredential

def setup_drivemm_azure():
    """设置DriveMM Azure ML环境"""
    
    # 从config.json读取配置
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 初始化Azure ML客户端
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name']
    )
    
    print("✅ Azure ML客户端连接成功")
    
    # 1. 创建GPU计算集群（如果不存在）
    print("🖥️ 检查GPU计算集群...")
    
    try:
        compute = ml_client.compute.get(config['compute_target'])
        print(f"✅ 找到现有集群: {compute.name}")
    except:
        print("⚠️ 创建新的GPU集群...")
        compute_config = AmlCompute(
            name=config['compute_target'],
            type="amlcompute",
            size="Standard_NC24ads_A100_v4",  # A100 GPU
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=1800
        )
        
        compute = ml_client.compute.begin_create_or_update(compute_config).result()
        print(f"✅ GPU集群创建成功: {compute.name}")
    
    # 2. 创建DriveMM环境
    print("🐳 创建DriveMM推理环境...")
    
    environment = Environment(
        name="drivemm-inference-env",
        description="DriveMM推理环境，支持GPU",
        conda_file="azure_drivemm_environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04:latest"
    )
    
    try:
        env = ml_client.environments.create_or_update(environment)
        print(f"✅ 环境创建成功: {env.name}")
    except Exception as e:
        print(f"⚠️ 环境可能已存在: {e}")
    
    # 3. 提交DriveMM推理作业
    print("🚀 提交DriveMM推理作业...")
    
    job = CommandJob(
        experiment_name=config['experiment_name'],
        display_name="DriveMM_Real_GPU_Inference",
        description="使用真实DriveMM模型在GPU上进行推理",
        compute=config['compute_target'],
        environment="drivemm-inference-env:latest",
        command="python azure_drivemm_real_inference.py",
        code="./",
        inputs={
            "storage_connection": os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        },
        outputs={
            "results": "./outputs/drivemm_results/"
        },
        resources={
            "instance_count": 1,
            "shm_size": "16g"
        },
        timeout=7200  # 2小时
    )
    
    try:
        submitted_job = ml_client.jobs.create_or_update(job)
        print(f"✅ 作业提交成功!")
        print(f"📊 作业名称: {submitted_job.name}")
        print(f"🔗 监控链接: {submitted_job.studio_url}")
        return submitted_job
    except Exception as e:
        print(f"❌ 作业提交失败: {e}")
        return None

if __name__ == "__main__":
    setup_drivemm_azure()
