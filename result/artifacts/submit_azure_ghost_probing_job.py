#!/usr/bin/env python3
"""
Azure ML A100 GPU Ghost Probing Job Submission Script
提交使用平衡版GPT-4.1的鬼探头检测作业到Azure ML
"""

import os
import sys
import logging
from datetime import datetime
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.core.exceptions import ClientAuthenticationError
import yaml

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureMLGhostProbingJobSubmitter:
    def __init__(self, 
                 subscription_id: str,
                 resource_group: str,
                 workspace_name: str,
                 compute_name: str = "gpu-cluster-a100"):
        """
        初始化Azure ML作业提交器
        
        Args:
            subscription_id: Azure订阅ID
            resource_group: 资源组名称
            workspace_name: 工作区名称
            compute_name: 计算集群名称
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.compute_name = compute_name
        
        # 尝试认证
        self.ml_client = self._authenticate()
        
        # 时间戳
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def _authenticate(self) -> MLClient:
        """认证并创建MLClient"""
        try:
            # 尝试默认凭据
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )
            
            # 测试连接
            workspace = ml_client.workspaces.get(self.workspace_name)
            logger.info(f"成功连接到工作区: {workspace.name}")
            
            return ml_client
            
        except ClientAuthenticationError:
            logger.warning("默认凭据失败，尝试交互式认证...")
            try:
                credential = InteractiveBrowserCredential()
                ml_client = MLClient(
                    credential=credential,
                    subscription_id=self.subscription_id,
                    resource_group_name=self.resource_group,
                    workspace_name=self.workspace_name
                )
                
                # 测试连接
                workspace = ml_client.workspaces.get(self.workspace_name)
                logger.info(f"成功连接到工作区: {workspace.name}")
                
                return ml_client
                
            except Exception as e:
                logger.error(f"认证失败: {e}")
                raise
    
    def create_environment(self) -> Environment:
        """创建或获取环境"""
        try:
            env_name = f"ghost-probing-gpt41-{self.timestamp}"
            
            # 检查环境文件是否存在
            env_file = "azure_ghost_probing_env.yml"
            if not os.path.exists(env_file):
                raise FileNotFoundError(f"环境文件不存在: {env_file}")
            
            # 创建环境
            environment = Environment(
                name=env_name,
                description="Ghost probing detection environment with GPT-4.1 support",
                conda_file=env_file,
                image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04:latest"
            )
            
            # 创建环境
            environment = self.ml_client.environments.create_or_update(environment)
            logger.info(f"环境已创建: {environment.name}")
            
            return environment
            
        except Exception as e:
            logger.error(f"创建环境失败: {e}")
            raise
    
    def upload_data(self):
        """上传数据到Azure ML数据存储"""
        try:
            from azure.ai.ml.entities import Data
            from azure.ai.ml.constants import AssetTypes
            
            # 上传视频数据
            if os.path.exists("DADA-2000-videos"):
                logger.info("上传视频数据...")
                video_data = Data(
                    name=f"dada-2000-videos-{self.timestamp}",
                    version="1",
                    description="DADA-2000 video dataset for ghost probing detection",
                    type=AssetTypes.URI_FOLDER,
                    path="DADA-2000-videos"
                )
                
                video_data = self.ml_client.data.create_or_update(video_data)
                logger.info(f"视频数据已上传: {video_data.name}")
            
            # 上传ground truth数据
            if os.path.exists("result/groundtruth_labels.csv"):
                logger.info("上传ground truth数据...")
                gt_data = Data(
                    name=f"groundtruth-labels-{self.timestamp}",
                    version="1",
                    description="Ground truth labels for ghost probing detection",
                    type=AssetTypes.URI_FILE,
                    path="result/groundtruth_labels.csv"
                )
                
                gt_data = self.ml_client.data.create_or_update(gt_data)
                logger.info(f"Ground truth数据已上传: {gt_data.name}")
                
        except Exception as e:
            logger.error(f"上传数据失败: {e}")
            raise
    
    def create_job(self, environment: Environment) -> command:
        """创建作业"""
        try:
            # 作业名称
            job_name = f"ghost-probing-gpt41-{self.timestamp}"
            
            # 创建命令作业
            job = command(
                name=job_name,
                display_name="Ghost Probing Detection with GPT-4.1 Balanced",
                description="Process 100 videos for ghost probing detection using balanced GPT-4.1 prompt",
                code=".",
                command="python batch_ghost_probing_gpt41_balanced.py --video-folder ./DADA-2000-videos --output-folder ./result --groundtruth-file ./result/groundtruth_labels.csv --max-videos 100",
                environment=environment,
                compute=self.compute_name,
                experiment_name="ghost_probing_detection",
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
                "VISION_DEPLOYMENT_NAME": os.getenv("VISION_DEPLOYMENT_NAME", "gpt-4.1"),
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
            
            logger.info(f"作业已创建: {job.name}")
            return job
            
        except Exception as e:
            logger.error(f"创建作业失败: {e}")
            raise
    
    def submit_job(self, job: command) -> str:
        """提交作业"""
        try:
            # 提交作业
            submitted_job = self.ml_client.jobs.create_or_update(job)
            
            logger.info(f"作业已提交: {submitted_job.name}")
            logger.info(f"作业状态: {submitted_job.status}")
            logger.info(f"作业链接: {submitted_job.studio_url}")
            
            return submitted_job.name
            
        except Exception as e:
            logger.error(f"提交作业失败: {e}")
            raise
    
    def monitor_job(self, job_name: str):
        """监控作业状态"""
        try:
            import time
            
            logger.info(f"开始监控作业: {job_name}")
            
            while True:
                job = self.ml_client.jobs.get(job_name)
                status = job.status
                
                logger.info(f"作业状态: {status}")
                
                if status in ["Completed", "Failed", "Canceled"]:
                    logger.info(f"作业已结束: {status}")
                    
                    if status == "Completed":
                        logger.info("作业成功完成！")
                        self.download_results(job_name)
                    else:
                        logger.error(f"作业失败或被取消: {status}")
                    
                    break
                
                # 等待30秒后再检查
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("监控已中断")
        except Exception as e:
            logger.error(f"监控作业失败: {e}")
    
    def download_results(self, job_name: str):
        """下载作业结果"""
        try:
            logger.info(f"开始下载作业结果: {job_name}")
            
            # 下载输出
            self.ml_client.jobs.download(
                name=job_name,
                download_path=f"./azure_ml_outputs/{job_name}"
            )
            
            logger.info(f"结果已下载到: ./azure_ml_outputs/{job_name}")
            
        except Exception as e:
            logger.error(f"下载结果失败: {e}")
    
    def run_complete_pipeline(self):
        """运行完整的提交流水线"""
        try:
            logger.info("=== 开始Azure ML作业提交流水线 ===")
            
            # 1. 创建环境
            logger.info("步骤 1: 创建环境")
            environment = self.create_environment()
            
            # 2. 上传数据
            logger.info("步骤 2: 上传数据")
            self.upload_data()
            
            # 3. 创建作业
            logger.info("步骤 3: 创建作业")
            job = self.create_job(environment)
            
            # 4. 提交作业
            logger.info("步骤 4: 提交作业")
            job_name = self.submit_job(job)
            
            # 5. 监控作业
            logger.info("步骤 5: 监控作业")
            self.monitor_job(job_name)
            
            logger.info("=== 流水线完成 ===")
            
        except Exception as e:
            logger.error(f"流水线执行失败: {e}")
            raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Submit Ghost Probing Detection Job to Azure ML')
    parser.add_argument('--subscription-id', required=True, help='Azure订阅ID')
    parser.add_argument('--resource-group', required=True, help='资源组名称')
    parser.add_argument('--workspace-name', required=True, help='工作区名称')
    parser.add_argument('--compute-name', default='gpu-cluster-a100', help='计算集群名称')
    parser.add_argument('--monitor-only', help='仅监控指定作业')
    parser.add_argument('--download-only', help='仅下载指定作业结果')
    
    args = parser.parse_args()
    
    # 检查环境变量
    required_env_vars = [
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_WHISPER_KEY',
        'AZURE_WHISPER_DEPLOYMENT',
        'AZURE_WHISPER_ENDPOINT'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"缺少必要的环境变量: {missing_vars}")
        logger.error("请设置这些环境变量后再运行")
        return
    
    # 创建提交器
    submitter = AzureMLGhostProbingJobSubmitter(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        compute_name=args.compute_name
    )
    
    try:
        if args.monitor_only:
            # 仅监控作业
            submitter.monitor_job(args.monitor_only)
        elif args.download_only:
            # 仅下载结果
            submitter.download_results(args.download_only)
        else:
            # 运行完整流水线
            submitter.run_complete_pipeline()
            
    except Exception as e:
        logger.error(f"执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()