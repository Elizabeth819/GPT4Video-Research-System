#!/usr/bin/env python3
"""
Azure ML LLaVA Ghost Probing Job Submission Script
提交LLaVA鬼探头检测作业到Azure ML
文件路径: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/submit_azure_llava_job.py
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

try:
    from azure.ai.ml import MLClient, command
    from azure.ai.ml.entities import Environment
    from azure.identity import DefaultAzureCredential
    from azure.core.exceptions import HttpResponseError
except ImportError:
    print("❌ Azure ML SDK未安装，请运行: pip install azure-ai-ml")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureLLaVAJobSubmitter:
    """Azure ML LLaVA作业提交器"""
    
    def __init__(self, 
                 subscription_id: str = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a",
                 resource_group: str = "llava-resourcegroup", 
                 workspace_name: str = "llava-workspace"):
        """
        初始化Azure ML客户端
        
        Args:
            subscription_id: Azure订阅ID
            resource_group: 资源组名称
            workspace_name: 工作区名称
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        
        try:
            # 初始化ML客户端
            credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
            
            logger.info(f"✅ Azure ML客户端初始化成功")
            logger.info(f"📋 订阅: {subscription_id}")
            logger.info(f"📋 资源组: {resource_group}")
            logger.info(f"📋 工作区: {workspace_name}")
            
        except Exception as e:
            logger.error(f"❌ Azure ML客户端初始化失败: {e}")
            raise
    
    def submit_llava_ghost_probing_job(self, 
                                      job_name: str = None,
                                      compute_name: str = "llava-a100-low-priority",
                                      environment_name: str = "AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10",
                                      video_limit: int = 100,
                                      save_interval: int = 5,
                                      dry_run: bool = False) -> str:
        """
        提交LLaVA鬼探头检测作业
        
        Args:
            job_name: 作业名称
            compute_name: 计算集群名称
            environment_name: 环境名称
            video_limit: 处理视频数量限制
            save_interval: 保存间隔
            dry_run: 是否为测试模式
            
        Returns:
            作业ID
        """
        try:
            # 生成作业名称
            if job_name is None:
                timestamp = datetime.now().strftime('%m%d_%H%M%S')
                job_name = f"llava-ghost-probing-{timestamp}"
            
            logger.info(f"🚀 准备提交LLaVA鬼探头检测作业: {job_name}")
            
            # 创建作业命令 - 使用干净的requirements文件
            command_str = f"""pip install -r requirements_clean.txt && python llava_ghost_probing_batch.py --video-folder ./inputs/video_data --output-folder ./outputs/llava_ghost_probing_results --limit {video_limit} --save-interval {save_interval}"""
            
            # 创建命令作业
            job = command(
                display_name=job_name,
                description=f"LLaVA-NeXT Ghost Probing Detection on {video_limit} DADA Videos",
                command=command_str,
                environment=f"azureml:{environment_name}",
                compute=compute_name,
                code=".",  # 包含当前目录的所有文件
                inputs={
                    "video_data": {
                        "type": "uri_folder",
                        "path": "azureml:DADA-100-videos:20250721_150147",
                        "mode": "ro_mount"
                    }
                },
                outputs={
                    "results": {
                        "type": "uri_folder", 
                        "path": f"azureml://datastores/workspaceblobstore/paths/llava-ghost-probing-results/{job_name}/",
                        "mode": "rw_mount"
                    }
                },
                environment_variables={
                    "CUDA_VISIBLE_DEVICES": "0",
                    "PYTHONPATH": "/mnt/batch/tasks/shared/LS_root/mounts/clusters/llava-a100-low-priority/code/Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA:/mnt/batch/tasks/shared/LS_root/mounts/clusters/llava-a100-low-priority/code/Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/LLaVA-NeXT",
                    "HF_HOME": "/tmp/huggingface",
                    "TORCH_HOME": "/tmp/torch",
                    "TRANSFORMERS_CACHE": "/tmp/transformers"
                },
                tags={
                    "model": "LLaVA-Video-7B-Qwen2",
                    "task": "ghost_probing_detection", 
                    "dataset": "DADA-100-videos",
                    "prompt": "balanced_gpt41_compatible"
                },
                experiment_name="llava-ghost-probing-experiment"
            )
            
            if dry_run:
                logger.info("🧪 Dry Run模式 - 作业配置验证")
                logger.info("✅ 作业配置验证通过")
                logger.info(f"📝 作业名称: {job_name}")
                logger.info(f"🖥️  计算集群: {compute_name}")
                logger.info(f"🎬 视频数量: {video_limit}")
                logger.info("💡 使用 --no-dry-run 提交真实作业")
                return f"dry-run-{job_name}"
            
            # 提交作业
            logger.info("📤 正在提交作业到Azure ML...")
            submitted_job = self.ml_client.jobs.create_or_update(job)
            
            logger.info(f"✅ 作业提交成功!")
            logger.info(f"🆔 作业ID: {submitted_job.name}")
            logger.info(f"📊 作业状态: {submitted_job.status}")
            logger.info(f"🔗 Azure ML Studio链接: {submitted_job.studio_url}")
            
            # 提供后续监控建议
            logger.info("\n💡 后续操作建议:")
            logger.info(f"   检查状态: python {__file__} --action status --job-name {submitted_job.name}")
            logger.info(f"   下载结果: python {__file__} --action download --job-name {submitted_job.name}")
            
            return submitted_job.name
            
        except HttpResponseError as e:
            logger.error(f"❌ Azure ML API错误: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 作业提交失败: {e}")
            raise
    
    def check_job_status(self, job_name: str):
        """检查作业状态"""
        try:
            job = self.ml_client.jobs.get(job_name)
            
            print(f"\n📊 作业状态报告: {job_name}")
            print("="*50)
            print(f"状态: {job.status}")
            print(f"开始时间: {job.creation_context.created_at}")
            print(f"Studio链接: {job.studio_url}")
            
            if job.status == "Completed":
                print("✅ 作业已完成!")
            elif job.status == "Failed":
                print("❌ 作业失败!")
                if hasattr(job, 'error'):
                    print(f"错误信息: {job.error}")
            elif job.status in ["Running", "Preparing", "Starting"]:
                print("🔄 作业正在运行中...")
            else:
                print(f"📋 当前状态: {job.status}")
            
            print("="*50)
            
        except Exception as e:
            logger.error(f"❌ 检查作业状态失败: {e}")
    
    def list_recent_jobs(self, limit: int = 10):
        """列出最近的作业"""
        try:
            jobs = list(self.ml_client.jobs.list(max_results=limit))
            
            print(f"\n📋 最近{len(jobs)}个作业:")
            print("="*80)
            print(f"{'作业名称':<30} {'状态':<12} {'创建时间':<20}")
            print("-"*80)
            
            for job in jobs:
                created_at = job.creation_context.created_at.strftime('%Y-%m-%d %H:%M:%S')
                print(f"{job.name:<30} {job.status:<12} {created_at:<20}")
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"❌ 列出作业失败: {e}")
    
    def download_job_outputs(self, job_name: str, local_path: str = "./downloaded_results"):
        """下载作业输出"""
        try:
            logger.info(f"📥 开始下载作业输出: {job_name}")
            
            # 创建本地目录
            local_path = Path(local_path)
            local_path.mkdir(parents=True, exist_ok=True)
            
            # 下载输出
            self.ml_client.jobs.download(
                name=job_name,
                download_path=local_path,
                output_name="results"
            )
            
            logger.info(f"✅ 作业输出下载完成: {local_path}")
            
        except Exception as e:
            logger.error(f"❌ 下载作业输出失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Azure ML LLaVA作业提交工具')
    parser.add_argument('--action', choices=['submit', 'status', 'list', 'download'], 
                       default='submit', help='执行的操作')
    parser.add_argument('--job-name', type=str, help='作业名称')
    parser.add_argument('--compute', type=str, default='llava-a100-low-priority',
                       help='计算集群名称')
    parser.add_argument('--limit', type=int, default=100,
                       help='处理视频数量限制')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='保存间隔')
    parser.add_argument('--download-path', type=str, default='./downloaded_results',
                       help='下载路径')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='测试模式，验证配置但不提交作业')
    parser.add_argument('--no-dry-run', action='store_true',
                       help='关闭测试模式，提交真实作业')
    
    args = parser.parse_args()
    
    try:
        # 创建提交器
        submitter = AzureLLaVAJobSubmitter()
        
        if args.action == 'submit':
            # 确定是否为dry run模式
            is_dry_run = args.dry_run and not args.no_dry_run
            
            # 提交作业
            job_id = submitter.submit_llava_ghost_probing_job(
                job_name=args.job_name,
                compute_name=args.compute,
                video_limit=args.limit,
                save_interval=args.save_interval,
                dry_run=is_dry_run
            )
            print(f"\n✅ 作业提交成功: {job_id}")
            print(f"💡 使用以下命令检查状态: python {__file__} --action status --job-name {job_id}")
            
        elif args.action == 'status':
            # 检查作业状态
            if not args.job_name:
                print("❌ 请指定作业名称: --job-name JOB_NAME")
                sys.exit(1)
            submitter.check_job_status(args.job_name)
            
        elif args.action == 'list':
            # 列出最近作业
            submitter.list_recent_jobs()
            
        elif args.action == 'download':
            # 下载作业输出
            if not args.job_name:
                print("❌ 请指定作业名称: --job-name JOB_NAME")
                sys.exit(1)
            submitter.download_job_outputs(args.job_name, args.download_path)
            
    except Exception as e:
        logger.error(f"❌ 操作失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()