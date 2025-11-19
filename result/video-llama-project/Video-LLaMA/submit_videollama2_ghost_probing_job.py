#!/usr/bin/env python3
"""
Submit Video-LLaMA2 Ghost Probing Detection Job to Azure ML
使用您现有的Azure ML环境提交Video-LLaMA2鬼探头检测作业
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

class VideoLLaMA2JobSubmitter:
    def __init__(self):
        self.subscription_id = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a"
        self.resource_group = "video-llama2-ghost-probing-rg"
        self.workspace_name = "video-llama2-ghost-probing-ws"
        self.compute_name = "gpu-cluster-a100"
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info(f"🚀 Video-LLaMA2 Ghost Probing Job Submitter")
        logger.info(f"   订阅ID: {self.subscription_id}")
        logger.info(f"   资源组: {self.resource_group}")
        logger.info(f"   工作区: {self.workspace_name}")
    
    def authenticate_azure_ml(self):
        """认证Azure ML"""
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
            
            return ml_client
            
        except Exception as e:
            logger.error(f"❌ Azure ML认证失败: {e}")
            return None
    
    def check_prerequisites(self):
        """检查先决条件"""
        logger.info("🔍 检查先决条件...")
        
        # 检查必要文件
        required_files = [
            "video_llama2_ghost_probing_detector.py",
            "video_llama2_environment.yml",
            "azure_ml_videollama2_ghost_probing_job.yml",
            "eval_configs/video_llama_eval_withaudio.yaml"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ 缺少必要文件: {missing_files}")
            return False
        
        logger.info("✅ 所有必要文件已找到")
        return True
    
    def upload_data_if_needed(self, ml_client):
        """如果需要，上传数据"""
        logger.info("📤 检查数据是否需要上传...")
        
        try:
            from azure.ai.ml.entities import Data
            from azure.ai.ml.constants import AssetTypes
            
            # 检查是否存在视频数据
            video_folder = Path("../../DADA-2000-videos")
            if video_folder.exists():
                logger.info("📹 本地找到DADA-2000视频数据")
                
                # 检查Azure ML中是否已存在数据
                try:
                    existing_data = ml_client.data.get("dada-2000-videos", version="latest")
                    logger.info(f"✅ Azure ML中已存在视频数据: {existing_data.name}")
                except:
                    logger.info("📤 上传视频数据到Azure ML...")
                    video_data = Data(
                        name="dada-2000-videos",
                        version="1",
                        description="DADA-2000 video dataset for ghost probing detection",
                        type=AssetTypes.URI_FOLDER,
                        path=str(video_folder)
                    )
                    ml_client.data.create_or_update(video_data)
                    logger.info("✅ 视频数据上传完成")
            
            # 检查ground truth文件
            gt_file = Path("../../result/groundtruth_labels.csv")
            if gt_file.exists():
                logger.info("📊 本地找到ground truth文件")
                
                try:
                    existing_gt = ml_client.data.get("groundtruth-labels", version="latest")
                    logger.info(f"✅ Azure ML中已存在ground truth数据: {existing_gt.name}")
                except:
                    logger.info("📤 上传ground truth数据到Azure ML...")
                    gt_data = Data(
                        name="groundtruth-labels",
                        version="1",
                        description="Ground truth labels for ghost probing detection",
                        type=AssetTypes.URI_FILE,
                        path=str(gt_file)
                    )
                    ml_client.data.create_or_update(gt_data)
                    logger.info("✅ Ground truth数据上传完成")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据上传失败: {e}")
            return False
    
    def create_environment(self, ml_client):
        """创建或更新环境"""
        logger.info("🐍 创建Video-LLaMA2环境...")
        
        try:
            from azure.ai.ml.entities import Environment
            
            environment = Environment(
                name=f"video-llama2-ghost-probing-{self.timestamp}",
                description="Video-LLaMA2 environment for ghost probing detection",
                conda_file="video_llama2_environment.yml",
                image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04:latest"
            )
            
            environment = ml_client.environments.create_or_update(environment)
            logger.info(f"✅ 环境创建完成: {environment.name}")
            
            return environment
            
        except Exception as e:
            logger.error(f"❌ 环境创建失败: {e}")
            return None
    
    def submit_job(self, ml_client, environment):
        """提交Video-LLaMA2作业"""
        logger.info("🚀 提交Video-LLaMA2 Ghost Probing检测作业...")
        
        try:
            from azure.ai.ml import command
            
            # 创建作业
            job = command(
                name=f"video-llama2-ghost-probing-{self.timestamp}",
                display_name=f"Video-LLaMA2 Ghost Probing Detection - {self.timestamp}",
                description="Video-LLaMA2 model for ghost probing detection on 100 DADA videos",
                code=".",
                command="python video_llama2_ghost_probing_detector.py --config eval_configs/video_llama_eval_withaudio.yaml --model-type llama_v2 --gpu-id 0 --video-folder ./DADA-2000-videos --groundtruth-file ./result/groundtruth_labels.csv --max-videos 100",
                environment=environment,
                compute=self.compute_name,
                experiment_name="video_llama2_ghost_probing",
                tags={
                    "model": "video-llama2",
                    "task": "ghost_probing_detection",
                    "dataset": "DADA-2000",
                    "video_count": "100",
                    "framework": "pytorch",
                    "gpu": "a100",
                    "timestamp": self.timestamp
                }
            )
            
            # 设置环境变量
            job.environment_variables = {
                "PYTHONPATH": "/mnt/batch/tasks/shared/LS_root/mounts/clusters/gpu-cluster-a100/code",
                "OMP_NUM_THREADS": "4",
                "CUDA_VISIBLE_DEVICES": "0",
                "TORCH_CUDA_ARCH_LIST": "8.0",
                "FORCE_CUDA": "1"
            }
            
            # 提交作业
            submitted_job = ml_client.jobs.create_or_update(job)
            
            logger.info(f"🎉 作业提交成功!")
            logger.info(f"   作业名称: {submitted_job.name}")
            logger.info(f"   作业ID: {submitted_job.id}")
            logger.info(f"   作业状态: {submitted_job.status}")
            logger.info(f"   Studio链接: {submitted_job.studio_url}")
            
            # 保存作业信息
            job_info = {
                "job_name": submitted_job.name,
                "job_id": submitted_job.id,
                "status": submitted_job.status,
                "studio_url": submitted_job.studio_url,
                "timestamp": self.timestamp,
                "model": "video-llama2",
                "task": "ghost_probing_detection",
                "submission_time": datetime.now().isoformat()
            }
            
            job_info_file = f"video_llama2_job_info_{self.timestamp}.json"
            with open(job_info_file, "w") as f:
                json.dump(job_info, f, indent=2)
            
            logger.info(f"📄 作业信息已保存: {job_info_file}")
            
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
                
                # 等待2分钟后再检查
                time.sleep(120)
                
        except KeyboardInterrupt:
            logger.info("⏹️ 监控已中断")
        except Exception as e:
            logger.error(f"❌ 监控失败: {e}")
    
    def download_results(self, ml_client, job_name):
        """下载作业结果"""
        logger.info("📥 下载作业结果...")
        
        try:
            output_dir = f"./video_llama2_outputs/{job_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            ml_client.jobs.download(
                name=job_name,
                download_path=output_dir
            )
            
            logger.info(f"✅ 结果已下载到: {output_dir}")
            
            # 查找并显示结果文件
            result_files = list(Path(output_dir).rglob("*.json")) + list(Path(output_dir).rglob("*.csv"))
            if result_files:
                logger.info("📄 找到结果文件:")
                for file in result_files:
                    logger.info(f"   📄 {file.name}")
            
        except Exception as e:
            logger.error(f"❌ 下载结果失败: {e}")
    
    def run_complete_pipeline(self):
        """运行完整的提交流水线"""
        logger.info("🔄 开始Video-LLaMA2 Ghost Probing检测流水线")
        
        try:
            # 1. 检查先决条件
            if not self.check_prerequisites():
                logger.error("❌ 先决条件检查失败")
                return False
            
            # 2. 认证Azure ML
            ml_client = self.authenticate_azure_ml()
            if ml_client is None:
                logger.error("❌ Azure ML认证失败")
                return False
            
            # 3. 上传数据
            if not self.upload_data_if_needed(ml_client):
                logger.error("❌ 数据上传失败")
                return False
            
            # 4. 创建环境
            environment = self.create_environment(ml_client)
            if environment is None:
                logger.error("❌ 环境创建失败")
                return False
            
            # 5. 提交作业
            submitted_job = self.submit_job(ml_client, environment)
            if submitted_job is None:
                logger.error("❌ 作业提交失败")
                return False
            
            # 6. 监控作业
            self.monitor_job(ml_client, submitted_job.name)
            
            logger.info("✅ 流水线执行完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 流水线执行失败: {e}")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Submit Video-LLaMA2 Ghost Probing Detection Job')
    parser.add_argument('--check-only', action='store_true', help='仅检查环境不提交作业')
    parser.add_argument('--no-monitor', action='store_true', help='提交作业但不监控')
    parser.add_argument('--monitor-only', help='仅监控指定作业')
    parser.add_argument('--download-only', help='仅下载指定作业结果')
    
    args = parser.parse_args()
    
    submitter = VideoLLaMA2JobSubmitter()
    
    if args.check_only:
        logger.info("🔍 仅检查环境...")
        prereq_ok = submitter.check_prerequisites()
        ml_client = submitter.authenticate_azure_ml()
        
        if prereq_ok and ml_client:
            logger.info("✅ 环境检查通过，可以提交作业")
        else:
            logger.error("❌ 环境检查失败")
        
    elif args.monitor_only:
        logger.info(f"👁️ 监控作业: {args.monitor_only}")
        ml_client = submitter.authenticate_azure_ml()
        if ml_client:
            submitter.monitor_job(ml_client, args.monitor_only)
        
    elif args.download_only:
        logger.info(f"📥 下载作业结果: {args.download_only}")
        ml_client = submitter.authenticate_azure_ml()
        if ml_client:
            submitter.download_results(ml_client, args.download_only)
        
    elif args.no_monitor:
        logger.info("🚀 提交作业但不监控...")
        if submitter.check_prerequisites():
            ml_client = submitter.authenticate_azure_ml()
            if ml_client:
                submitter.upload_data_if_needed(ml_client)
                environment = submitter.create_environment(ml_client)
                if environment:
                    submitter.submit_job(ml_client, environment)
    else:
        # 运行完整流水线
        submitter.run_complete_pipeline()


if __name__ == "__main__":
    main()