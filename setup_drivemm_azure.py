#!/usr/bin/env python
"""
设置DriveMM在Azure ML上的推理环境并提交作业
"""

import os
import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, CommandJob, AmlCompute
from azure.identity import DefaultAzureCredential

def setup_drivemm_azure():
    """设置DriveMM Azure ML环境并提交作业"""
    
    # 从config.json读取配置
    if not os.path.exists('config.json'):
        print("❌ 找不到config.json文件")
        print("📝 请复制 config.json.example 为 config.json 并填入实际配置")
        return None
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 初始化Azure ML客户端
    print("🔗 连接到Azure ML...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name']
    )
    
    print("✅ Azure ML客户端连接成功")
    
    # 1. 检查GPU计算集群
    print("🖥️  检查GPU计算集群...")
    
    try:
        compute = ml_client.compute.get(config.get('compute_target', 'drivemm-a100-cluster'))
        print(f"✅ 找到现有集群: {compute.name}")
        print(f"   类型: {compute.size}")
        print(f"   状态: {compute.provisioning_state}")
    except Exception as e:
        print(f"⚠️  集群不存在或无法访问: {e}")
        print("💡 你可以在Azure ML Studio中创建集群，或使用以下代码创建:")
        print("""
        compute_config = AmlCompute(
            name="drivemm-a100-cluster",
            type="amlcompute",
            size="Standard_NC24ads_A100_v4",  # A100 40GB GPU
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=1800
        )
        compute = ml_client.compute.begin_create_or_update(compute_config).result()
        """)
        return None
    
    # 2. 创建或更新DriveMM环境
    print("🐳 创建DriveMM推理环境...")
    
    environment = Environment(
        name="drivemm-inference-env",
        description="DriveMM推理环境，支持GPU",
        conda_file="azure_drivemm_environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04:latest"
    )
    
    try:
        env = ml_client.environments.create_or_update(environment)
        print(f"✅ 环境创建/更新成功: {env.name}:{env.version}")
    except Exception as e:
        print(f"⚠️  环境创建失败: {e}")
        return None
    
    # 3. 提交DriveMM推理作业
    print("🚀 提交DriveMM推理作业...")
    
    # 获取存储连接字符串
    storage_connection = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    if not storage_connection:
        print("⚠️  警告: 未设置AZURE_STORAGE_CONNECTION_STRING环境变量")
        print("   请运行: export AZURE_STORAGE_CONNECTION_STRING='你的连接字符串'")
    
    job = CommandJob(
        experiment_name=config.get('experiment_name', 'drivemm-inference'),
        display_name="DriveMM_Real_GPU_Inference",
        description="使用真实DriveMM模型在A100 GPU上对dada-videos进行推理",
        compute=config.get('compute_target', 'drivemm-a100-cluster'),
        environment=f"{env.name}:{env.version}",
        command="python azure_drivemm_real_inference.py",
        code="./",
        environment_variables={
            "AZURE_STORAGE_CONNECTION_STRING": storage_connection
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
        print(f"\n💡 使用以下命令查看日志:")
        print(f"   az ml job stream --name {submitted_job.name} \\")
        print(f"      --workspace-name {config['workspace_name']} \\")
        print(f"      --resource-group {config['resource_group']}")
        return submitted_job
    except Exception as e:
        print(f"❌ 作业提交失败: {e}")
        return None

if __name__ == "__main__":
    print("🔧 Azure ML DriveMM推理环境设置")
    print("=" * 60)
    setup_drivemm_azure()
