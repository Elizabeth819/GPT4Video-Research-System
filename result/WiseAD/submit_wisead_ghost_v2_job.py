#!/usr/bin/env python3
"""
提交WiseAD v2.0鬼探头检测作业到Azure ML
10秒段检测模式，确保输出保存
"""

import json
import logging
from datetime import datetime
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def submit_wisead_v2_job():
    """提交WiseAD v2.0鬼探头检测作业"""
    
    try:
        # 加载配置
        with open('wisead_ghost_probing_v2_config.json', 'r') as f:
            config = json.load(f)
        
        logger.info("🚀 开始提交WiseAD v2.0鬼探头检测作业")
        logger.info(f"📋 作业配置: {config['display_name']}")
        
        # 初始化Azure ML客户端 - 使用正确的wisead配置
        credential = DefaultAzureCredential()
        subscription_id = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a"  # wisead订阅ID
        resource_group = "wisead-rg"  # 正确的wisead资源组
        workspace_name = "wisead-ml-workspace"  # 正确的wisead工作区
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        logger.info("✅ Azure ML客户端初始化成功 (wisead-rg)")
        
        # 创建作业
        job = command(
            display_name=config["display_name"],
            description=config["description"],
            experiment_name=config["experiment_name"],
            command="python azure_ml_wisead_ghost_probing_v2.py",
            environment="AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10",  # 正确的环境
            compute=config["compute"],
            tags=config["tags"]
        )
        
        # 提交作业
        logger.info("📤 正在提交作业到Azure ML...")
        submitted_job = ml_client.jobs.create_or_update(job)
        
        logger.info("✅ WiseAD v2.0作业提交成功!")
        logger.info(f"🆔 作业ID: {submitted_job.name}")
        logger.info(f"🔗 作业URL: https://ml.azure.com/runs/{submitted_job.name}")
        logger.info(f"🎯 计算节点: {config['compute']}")
        logger.info(f"⏱️ 超时设置: {config['settings']['timeout']}")
        
        # 保存作业信息
        job_info = {
            "job_id": submitted_job.name,
            "job_url": f"https://ml.azure.com/runs/{submitted_job.name}",
            "submit_time": datetime.now().isoformat(),
            "config": config,
            "status": "submitted",
            "azure_config": {
                "subscription_id": subscription_id,
                "resource_group": resource_group,
                "workspace": workspace_name
            }
        }
        
        with open(f"wisead_v2_job_{submitted_job.name}.json", 'w') as f:
            json.dump(job_info, f, indent=2)
        
        logger.info(f"📋 作业信息已保存: wisead_v2_job_{submitted_job.name}.json")
        
        print("\n" + "="*80)
        print("🎉 WiseAD v2.0 鬼探头检测作业提交成功!")
        print("="*80)
        print(f"📊 作业详情:")
        print(f"   - 作业ID: {submitted_job.name}")
        print(f"   - 实验名: {config['experiment_name']}")
        print(f"   - 计算节点: {config['compute']}")
        print(f"   - 资源组: {resource_group}")
        print(f"   - 工作区: {workspace_name}")
        print(f"   - 检测模式: 10秒段模式")
        print(f"   - 预期输出: 详细鬼探头日志 + JSON结果文件")
        print(f"   - 监控URL: https://ml.azure.com/runs/{submitted_job.name}")
        print("="*80)
        
        return submitted_job.name
        
    except Exception as e:
        logger.error(f"❌ 作业提交失败: {e}")
        return None

def main():
    """主函数"""
    job_id = submit_wisead_v2_job()
    
    if job_id:
        print(f"\n✅ WiseAD v2.0作业提交成功! 作业ID: {job_id}")
        print(f"🔍 使用以下命令监控作业状态:")
        print(f"   az ml job show -n {job_id}")
        print(f"📥 作业完成后可下载输出文件")
    else:
        print("❌ WiseAD v2.0作业提交失败")

if __name__ == "__main__":
    main() 