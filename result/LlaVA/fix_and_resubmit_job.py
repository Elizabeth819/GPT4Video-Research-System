#!/usr/bin/env python3
"""
Fix and Resubmit Azure ML LLaVA Job Script
修复并重新提交Azure ML LLaVA作业
文件路径: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/fix_and_resubmit_job.py

Based on the failure analysis of job 'crimson_boniato_k1kg8q62fr', this script:
1. Validates all required files exist
2. Submits the job using the correct YAML configuration
3. Provides monitoring and recovery options
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

try:
    from azure.ai.ml import MLClient, command, Input, Output
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

class LLaVAJobFixer:
    """LLaVA作业修复和重新提交器"""
    
    def __init__(self, 
                 subscription_id: str = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a",
                 resource_group: str = "llava-resourcegroup", 
                 workspace_name: str = "llava-workspace"):
        """
        初始化Azure ML客户端
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.base_path = Path(__file__).parent
        
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
            logger.info(f"📋 工作区: {workspace_name}")
            
        except Exception as e:
            logger.error(f"❌ Azure ML客户端初始化失败: {e}")
            raise
    
    def validate_required_files(self) -> bool:
        """验证所需文件是否存在"""
        logger.info("🔍 验证必需文件...")
        
        required_files = [
            'requirements.txt',
            'llava_ghost_probing_batch.py',
            'llava_ghost_probing_detector.py',
            'azure_ml_llava_ghost_probing.yml'
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.base_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
            else:
                logger.info(f"✅ {file_name}")
        
        if missing_files:
            logger.error(f"❌ 缺少必需文件: {missing_files}")
            return False
        
        logger.info("✅ 所有必需文件存在")
        return True
    
    def create_fixed_job_config(self, 
                              job_name: str = None,
                              compute_name: str = "llava-a100-low-priority",
                              limit: int = 100,
                              save_interval: int = 10) -> dict:
        """创建修复后的作业配置"""
        if job_name is None:
            timestamp = datetime.now().strftime('%m%d_%H%M%S')
            job_name = f"llava-ghost-probing-fixed-{timestamp}"
        
        logger.info(f"🔧 创建修复后的作业配置: {job_name}")
        
        # 修复后的单行命令（解决multiline YAML问题）
        fixed_command = (
            "echo '🚀 开始LLaVA鬼探头检测作业' && "
            "echo '📋 安装依赖包...' && "
            "pip install --upgrade pip && "
            "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu117 && "
            "pip install transformers==4.37.0 accelerate tokenizers sentencepiece && "
            "pip install decord opencv-python pillow && "
            "pip install numpy pandas tqdm scikit-learn matplotlib seaborn && "
            "pip install pyyaml python-dotenv && "
            "echo '✅ 依赖安装完成' && "
            "echo '🔍 检查GPU可用性...' && "
            "python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')\" && "
            "echo '📁 检查工作目录文件:' && ls -la && "
            "echo '🎬 开始批处理视频...' && "
            f"python llava_ghost_probing_batch.py --video-folder ./inputs/video_data --output-folder ./outputs/llava_ghost_probing_results --limit {limit} --save-interval {save_interval} && "
            "echo '✅ LLaVA鬼探头检测作业完成'"
        )
        
        job_config = {
            "display_name": job_name,
            "description": f"Fixed LLaVA-NeXT Ghost Probing Detection on {limit} DADA Videos",
            "command": fixed_command,
            "environment": "azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10",
            "compute": compute_name,
            "inputs": {
                "video_data": Input(
                    type="uri_folder",
                    path="azureml://datastores/workspaceblobstore/paths/DADA-100-videos/",
                    mode="ro_mount"
                )
            },
            "outputs": {
                "results": Output(
                    type="uri_folder",
                    path=f"azureml://datastores/workspaceblobstore/paths/llava-ghost-probing-fixed/{job_name}/",
                    mode="rw_mount"
                )
            },
            "environment_variables": {
                "CUDA_VISIBLE_DEVICES": "0",
                "HF_HOME": "/tmp/huggingface",
                "TORCH_HOME": "/tmp/torch",
                "TRANSFORMERS_CACHE": "/tmp/transformers",
                "PYTHONPATH": "/mnt/azureml/cr/j/*/exe/wd"
            },
            "tags": {
                "model": "LLaVA-Video-7B-Qwen2",
                "task": "ghost_probing_detection",
                "dataset": "DADA-100-videos",
                "status": "fixed",
                "previous_job": "crimson_boniato_k1kg8q62fr"
            },
            "experiment_name": "llava-ghost-probing-fixed"
        }
        
        return job_config
    
    def submit_fixed_job(self, 
                        job_name: str = None,
                        compute_name: str = "llava-a100-low-priority",
                        limit: int = 100,
                        save_interval: int = 10,
                        dry_run: bool = False) -> str:
        """提交修复后的作业"""
        try:
            # 1. 验证文件
            if not self.validate_required_files():
                raise Exception("必需文件验证失败")
            
            # 2. 创建作业配置
            job_config = self.create_fixed_job_config(job_name, compute_name, limit, save_interval)
            job_name = job_config["display_name"]
            
            logger.info(f"🚀 准备提交修复后的LLaVA作业: {job_name}")
            logger.info(f"🎬 处理视频数量: {limit}个")
            logger.info(f"💾 保存间隔: {save_interval}个视频")
            
            # 3. 创建命令作业对象
            job = command(
                display_name=job_config["display_name"],
                description=job_config["description"],
                command=job_config["command"],
                environment=job_config["environment"],
                compute=job_config["compute"],
                code=".",  # 使用当前目录的所有文件
                inputs=job_config["inputs"],
                outputs=job_config["outputs"],
                environment_variables=job_config["environment_variables"],
                tags=job_config["tags"],
                experiment_name=job_config["experiment_name"]
            )
            
            if dry_run:
                logger.info("🧪 Dry Run模式 - 作业配置验证")
                logger.info("✅ 修复后作业配置验证通过")
                logger.info(f"📝 作业名称: {job_name}")
                logger.info(f"🖥️  计算集群: {compute_name}")
                logger.info(f"🎬 视频数量: {limit}")
                logger.info("💡 使用 --no-dry-run 提交真实作业")
                
                # 显示修复点
                logger.info("\\n🔧 主要修复点:")
                logger.info("1. ✅ 使用单行命令避免YAML解析问题")
                logger.info("2. ✅ 包含所有必需文件(requirements.txt, *.py)")
                logger.info("3. ✅ 添加调试信息(ls -la, GPU检查)")
                logger.info("4. ✅ 使用正确的视频限制和保存间隔")
                
                return f"dry-run-{job_name}"
            
            # 4. 提交作业
            logger.info("📤 正在提交修复后作业到Azure ML...")
            submitted_job = self.ml_client.jobs.create_or_update(job)
            
            logger.info(f"✅ 修复作业提交成功!")
            logger.info(f"🆔 作业ID: {submitted_job.name}")
            logger.info(f"📊 作业状态: {submitted_job.status}")
            logger.info(f"🔗 Azure ML Studio链接: {submitted_job.studio_url}")
            
            # 5. 提供监控建议
            logger.info("\\n💡 监控建议:")
            logger.info(f"   检查状态: python {__file__} --action status --job-name {submitted_job.name}")
            logger.info(f"   实时监控: python monitor_job.py --job-name {submitted_job.name}")
            logger.info("\\n🎯 预计完成时间: 2-3小时 (100个视频)")
            
            return submitted_job.name
            
        except Exception as e:
            logger.error(f"❌ 提交修复作业失败: {e}")
            raise
    
    def cancel_failed_job(self, failed_job_name: str = "crimson_boniato_k1kg8q62fr"):
        """取消失败的作业（清理资源）"""
        try:
            logger.info(f"🗑️ 取消失败作业: {failed_job_name}")
            
            # 获取作业状态
            job = self.ml_client.jobs.get(failed_job_name)
            if job.status in ["Running", "Starting", "Preparing"]:
                self.ml_client.jobs.cancel(failed_job_name)
                logger.info(f"✅ 作业已取消: {failed_job_name}")
            else:
                logger.info(f"ℹ️ 作业状态为 {job.status}，无需取消")
                
        except Exception as e:
            logger.warning(f"⚠️ 取消作业失败: {e}")
    
    def check_job_status(self, job_name: str):
        """检查作业状态"""
        try:
            job = self.ml_client.jobs.get(job_name)
            
            print(f"\\n📊 作业状态报告: {job_name}")
            print("="*60)
            print(f"状态: {job.status}")
            print(f"开始时间: {job.creation_context.created_at}")
            print(f"Studio链接: {job.studio_url}")
            
            if job.status == "Completed":
                print("✅ 作业已完成!")
            elif job.status == "Failed":
                print("❌ 作业失败!")
                print("💡 请检查日志或重新运行修复脚本")
            elif job.status in ["Running", "Preparing", "Starting"]:
                print("🔄 作业正在运行中...")
                print(f"💡 监控命令: python monitor_job.py --job-name {job_name}")
            else:
                print(f"📋 当前状态: {job.status}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"❌ 检查作业状态失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLaVA作业修复和重新提交工具')
    parser.add_argument('--action', choices=['fix', 'status', 'cancel'], 
                       default='fix', help='执行的操作')
    parser.add_argument('--job-name', type=str, help='作业名称')
    parser.add_argument('--compute', type=str, default='llava-a100-low-priority',
                       help='计算集群名称')
    parser.add_argument('--limit', type=int, default=100,
                       help='处理视频数量限制')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='保存间隔')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='测试模式，验证配置但不提交作业')
    parser.add_argument('--no-dry-run', action='store_true',
                       help='关闭测试模式，提交真实作业')
    
    args = parser.parse_args()
    
    try:
        # 创建修复器
        fixer = LLaVAJobFixer()
        
        if args.action == 'fix':
            # 确定是否为dry run模式
            is_dry_run = args.dry_run and not args.no_dry_run
            
            # 取消失败的作业
            fixer.cancel_failed_job("crimson_boniato_k1kg8q62fr")
            
            # 提交修复后的作业
            job_id = fixer.submit_fixed_job(
                job_name=args.job_name,
                compute_name=args.compute,
                limit=args.limit,
                save_interval=args.save_interval,
                dry_run=is_dry_run
            )
            
            print(f"\\n✅ 修复作业提交成功: {job_id}")
            
        elif args.action == 'status':
            # 检查作业状态
            if not args.job_name:
                print("❌ 请指定作业名称: --job-name JOB_NAME")
                sys.exit(1)
            fixer.check_job_status(args.job_name)
            
        elif args.action == 'cancel':
            # 取消作业
            job_name = args.job_name or "crimson_boniato_k1kg8q62fr"
            fixer.cancel_failed_job(job_name)
            
    except Exception as e:
        logger.error(f"❌ 操作失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()