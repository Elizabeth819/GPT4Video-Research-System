#!/usr/bin/env python3
"""
Azure ML GPT-4.1 Balanced Ghost Probing Job Submission
使用您的Azure ML环境提交GPT-4.1平衡版鬼探头检测作业
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# 设置Azure ML环境变量
os.environ["AZURE_SUBSCRIPTION_ID"] = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a"
os.environ["AZURE_RESOURCE_GROUP"] = "video-llama2-ghost-probing-rg"
os.environ["AZURE_WORKSPACE_NAME"] = "video-llama2-ghost-probing-ws"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPT41BalancedJobSubmitter:
    def __init__(self):
        self.subscription_id = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a"
        self.resource_group = "video-llama2-ghost-probing-rg"
        self.workspace_name = "video-llama2-ghost-probing-ws"
        self.compute_name = "gpu-cluster-a100"  # 默认计算集群名
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info(f"初始化Azure ML客户端:")
        logger.info(f"  订阅ID: {self.subscription_id}")
        logger.info(f"  资源组: {self.resource_group}")
        logger.info(f"  工作区: {self.workspace_name}")
    
    def check_environment(self):
        """检查本地环境"""
        logger.info("🔍 检查本地环境...")
        
        # 检查必要的文件
        required_files = [
            "batch_ghost_probing_gpt41_balanced.py",
            "azure_ghost_probing_env.yml",
            "result/groundtruth_labels.csv"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ 缺少必要文件: {missing_files}")
            return False
        
        # 检查视频文件
        video_folder = Path("DADA-2000-videos")
        if not video_folder.exists():
            logger.error("❌ 视频文件夹不存在: DADA-2000-videos")
            return False
        
        # 统计目标视频
        target_videos = []
        for i in range(1, 6):  # images_1_* 到 images_5_*
            pattern = f"images_{i}_*.avi"
            videos = sorted(video_folder.glob(pattern))
            target_videos.extend(videos)
            if len(target_videos) >= 100:
                break
        
        target_videos = target_videos[:100]
        logger.info(f"✅ 找到 {len(target_videos)} 个目标视频")
        
        if len(target_videos) < 100:
            logger.warning(f"⚠️ 视频数量不足100个")
        
        return True
    
    def check_azure_credentials(self):
        """检查Azure凭据"""
        logger.info("🔑 检查Azure凭据...")
        
        try:
            from azure.identity import DefaultAzureCredential
            from azure.ai.ml import MLClient
            
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )
            
            # 测试连接
            workspace = ml_client.workspaces.get(self.workspace_name)
            logger.info(f"✅ 成功连接到Azure ML工作区: {workspace.name}")
            
            # 检查计算资源
            try:
                compute = ml_client.compute.get(self.compute_name)
                logger.info(f"✅ 找到计算资源: {compute.name} ({compute.type})")
            except Exception as e:
                logger.warning(f"⚠️ 计算资源 {self.compute_name} 不存在: {e}")
                logger.info("将使用默认计算资源或创建新的")
            
            return ml_client
            
        except Exception as e:
            logger.error(f"❌ Azure认证失败: {e}")
            return None
    
    def create_job_config(self):
        """创建作业配置"""
        logger.info("📝 创建作业配置...")
        
        job_config = {
            "$schema": "https://azuremlschemas.azureedge.net/latest/commandJob.schema.json",
            "type": "command",
            "display_name": f"gpt41-balanced-ghost-probing-{self.timestamp}",
            "experiment_name": "ghost_probing_gpt41_balanced",
            "description": "Ghost probing detection using GPT-4.1 balanced prompt on 100 DADA videos",
            "code": ".",
            "environment": {
                "conda_file": "azure_ghost_probing_env.yml",
                "image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04:latest"
            },
            "compute": self.compute_name,
            "command": "python batch_ghost_probing_gpt41_balanced.py --video-folder ./DADA-2000-videos --output-folder ./outputs --groundtruth-file ./result/groundtruth_labels.csv --max-videos 100",
            "environment_variables": {
                "AZURE_OPENAI_API_KEY": "${{secrets.AZURE_OPENAI_API_KEY}}",
                "AZURE_OPENAI_ENDPOINT": "${{secrets.AZURE_OPENAI_ENDPOINT}}",
                "VISION_API_TYPE": "Azure",
                "VISION_DEPLOYMENT_NAME": "gpt-4.1",
                "VISION_ENDPOINT": "${{secrets.AZURE_OPENAI_ENDPOINT}}",
                "OPENAI_API_VERSION": "2024-02-15-preview",
                "AUDIO_API_TYPE": "Azure",
                "AZURE_WHISPER_KEY": "${{secrets.AZURE_WHISPER_KEY}}",
                "AZURE_WHISPER_DEPLOYMENT": "${{secrets.AZURE_WHISPER_DEPLOYMENT}}",
                "AZURE_WHISPER_ENDPOINT": "${{secrets.AZURE_WHISPER_ENDPOINT}}",
                "PYTHONPATH": "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-cluster-a100/code",
                "OMP_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "0"
            },
            "settings": {
                "timeout": 14400,  # 4小时
                "priority": "high"
            },
            "tags": {
                "model": "gpt-4.1-balanced",
                "task": "ghost_probing_detection",
                "dataset": "DADA-2000",
                "video_count": "100",
                "timestamp": self.timestamp
            }
        }
        
        # 保存配置文件
        config_file = f"job_config_{self.timestamp}.yml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(job_config, f, default_flow_style=False)
        
        logger.info(f"✅ 作业配置已保存: {config_file}")
        return job_config
    
    def submit_job(self, ml_client):
        """提交作业到Azure ML"""
        logger.info("🚀 提交作业到Azure ML...")
        
        try:
            from azure.ai.ml import command
            from azure.ai.ml.entities import Environment
            
            # 创建环境
            environment = Environment(
                name=f"ghost-probing-gpt41-{self.timestamp}",
                description="Ghost probing detection environment with GPT-4.1 support",
                conda_file="azure_ghost_probing_env.yml",
                image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04:latest"
            )
            
            # 创建并提交环境
            logger.info("📦 创建环境...")
            environment = ml_client.environments.create_or_update(environment)
            
            # 创建命令作业
            job = command(
                name=f"gpt41-balanced-{self.timestamp}",
                display_name=f"GPT-4.1 Balanced Ghost Probing - {self.timestamp}",
                description="Process 100 DADA videos for ghost probing detection using balanced GPT-4.1 prompt",
                code=".",
                command="python batch_ghost_probing_gpt41_balanced.py --video-folder ./DADA-2000-videos --output-folder ./outputs --groundtruth-file ./result/groundtruth_labels.csv --max-videos 100",
                environment=environment,
                compute=self.compute_name,
                experiment_name="ghost_probing_gpt41_balanced",
                tags={
                    "model": "gpt-4.1-balanced",
                    "task": "ghost_probing_detection", 
                    "dataset": "DADA-2000",
                    "video_count": "100",
                    "timestamp": self.timestamp
                }
            )
            
            # 设置环境变量
            job.environment_variables = {
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", ""),
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                "VISION_API_TYPE": "Azure",
                "VISION_DEPLOYMENT_NAME": "gpt-4.1",
                "VISION_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                "OPENAI_API_VERSION": "2024-02-15-preview",
                "AUDIO_API_TYPE": "Azure",
                "AZURE_WHISPER_KEY": os.getenv("AZURE_WHISPER_KEY", ""),
                "AZURE_WHISPER_DEPLOYMENT": os.getenv("AZURE_WHISPER_DEPLOYMENT", ""),
                "AZURE_WHISPER_ENDPOINT": os.getenv("AZURE_WHISPER_ENDPOINT", ""),
                "PYTHONPATH": "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-cluster-a100/code",
                "OMP_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "0"
            }
            
            # 提交作业
            submitted_job = ml_client.jobs.create_or_update(job)
            
            logger.info(f"✅ 作业已提交成功!")
            logger.info(f"   作业名称: {submitted_job.name}")
            logger.info(f"   作业状态: {submitted_job.status}")
            logger.info(f"   Studio链接: {submitted_job.studio_url}")
            
            # 保存作业信息
            job_info = {
                "job_name": submitted_job.name,
                "job_id": submitted_job.id,
                "status": submitted_job.status,
                "studio_url": submitted_job.studio_url,
                "timestamp": self.timestamp,
                "subscription_id": self.subscription_id,
                "resource_group": self.resource_group,
                "workspace_name": self.workspace_name
            }
            
            with open(f"job_info_{self.timestamp}.json", "w") as f:
                json.dump(job_info, f, indent=2)
            
            return submitted_job
            
        except Exception as e:
            logger.error(f"❌ 作业提交失败: {e}")
            return None
    
    def monitor_job(self, ml_client, job_name):
        """监控作业状态"""
        logger.info(f"👁️ 开始监控作业: {job_name}")
        
        try:
            import time
            
            while True:
                job = ml_client.jobs.get(job_name)
                status = job.status
                
                logger.info(f"📊 作业状态: {status}")
                
                if status in ["Completed", "Failed", "Canceled"]:
                    if status == "Completed":
                        logger.info("🎉 作业成功完成!")
                        self.download_results(ml_client, job_name)
                    else:
                        logger.error(f"❌ 作业结束: {status}")
                    break
                
                # 等待60秒后再检查
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("⏹️ 监控已中断")
        except Exception as e:
            logger.error(f"❌ 监控失败: {e}")
    
    def download_results(self, ml_client, job_name):
        """下载作业结果"""
        logger.info("📥 下载作业结果...")
        
        try:
            output_dir = f"./azure_outputs/{job_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            ml_client.jobs.download(
                name=job_name,
                download_path=output_dir
            )
            
            logger.info(f"✅ 结果已下载到: {output_dir}")
            
            # 查找结果文件
            result_files = list(Path(output_dir).rglob("*.json")) + list(Path(output_dir).rglob("*.csv"))
            if result_files:
                logger.info("📄 找到结果文件:")
                for file in result_files:
                    logger.info(f"   {file}")
            
        except Exception as e:
            logger.error(f"❌ 下载结果失败: {e}")
    
    def run_complete_pipeline(self):
        """运行完整的提交和监控流水线"""
        logger.info("🚀 开始GPT-4.1 Balanced Ghost Probing作业提交流水线")
        
        # 1. 检查本地环境
        if not self.check_environment():
            logger.error("❌ 本地环境检查失败")
            return False
        
        # 2. 检查Azure凭据
        ml_client = self.check_azure_credentials()
        if ml_client is None:
            logger.error("❌ Azure凭据检查失败")
            return False
        
        # 3. 创建作业配置
        job_config = self.create_job_config()
        
        # 4. 提交作业
        submitted_job = self.submit_job(ml_client)
        if submitted_job is None:
            logger.error("❌ 作业提交失败")
            return False
        
        # 5. 监控作业
        self.monitor_job(ml_client, submitted_job.name)
        
        logger.info("✅ 流水线完成")
        return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Submit GPT-4.1 Balanced Ghost Probing Job')
    parser.add_argument('--compute-name', default='gpu-cluster-a100', help='计算集群名称')
    parser.add_argument('--no-monitor', action='store_true', help='提交后不监控')
    parser.add_argument('--check-only', action='store_true', help='仅检查环境不提交')
    
    args = parser.parse_args()
    
    submitter = GPT41BalancedJobSubmitter()
    submitter.compute_name = args.compute_name
    
    if args.check_only:
        logger.info("🔍 仅检查环境...")
        env_ok = submitter.check_environment()
        ml_client = submitter.check_azure_credentials()
        if env_ok and ml_client:
            logger.info("✅ 环境检查通过，可以提交作业")
        else:
            logger.error("❌ 环境检查失败")
        return
    
    if args.no_monitor:
        logger.info("🚀 提交作业但不监控...")
        if submitter.check_environment():
            ml_client = submitter.check_azure_credentials()
            if ml_client:
                submitter.submit_job(ml_client)
    else:
        submitter.run_complete_pipeline()

if __name__ == "__main__":
    main()