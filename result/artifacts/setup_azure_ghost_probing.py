#!/usr/bin/env python3
"""
Azure ML Ghost Probing Detection 环境设置脚本
帮助用户快速设置Azure ML环境用于鬼探头检测
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureMLGhostProbingSetup:
    def __init__(self):
        self.required_files = [
            "batch_ghost_probing_gpt41_balanced.py",
            "azure_ghost_probing_env.yml",
            "azure_ml_ghost_probing_gpt41_config.yml",
            "submit_azure_ghost_probing_job.py",
            "ActionSummary-gpt41-balanced-prompt.py",
            "BALANCED_GPT41_PROMPT_FINAL.md"
        ]
        
        self.required_env_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT", 
            "VISION_DEPLOYMENT_NAME",
            "AZURE_WHISPER_KEY",
            "AZURE_WHISPER_DEPLOYMENT",
            "AZURE_WHISPER_ENDPOINT"
        ]
        
        self.optional_env_vars = [
            "AZURE_SUBSCRIPTION_ID",
            "AZURE_RESOURCE_GROUP",
            "AZURE_WORKSPACE_NAME",
            "AZURE_COMPUTE_NAME"
        ]
    
    def check_dependencies(self) -> bool:
        """检查依赖项"""
        logger.info("检查依赖项...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
            logger.error(f"需要Python 3.9或更高版本，当前版本: {python_version.major}.{python_version.minor}")
            return False
        
        # 检查必要的包
        required_packages = [
            "azure-ai-ml",
            "azure-identity", 
            "openai",
            "pandas",
            "numpy",
            "opencv-python",
            "moviepy",
            "tqdm"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"缺少必要的包: {missing_packages}")
            logger.info("请运行: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("依赖项检查通过")
        return True
    
    def check_files(self) -> bool:
        """检查必要的文件"""
        logger.info("检查必要的文件...")
        
        missing_files = []
        for file in self.required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"缺少必要的文件: {missing_files}")
            return False
        
        logger.info("文件检查通过")
        return True
    
    def check_environment_variables(self) -> Dict[str, str]:
        """检查环境变量"""
        logger.info("检查环境变量...")
        
        env_status = {}
        
        # 检查必要的环境变量
        for var in self.required_env_vars:
            value = os.getenv(var)
            if value:
                env_status[var] = "✓ 已设置"
            else:
                env_status[var] = "✗ 未设置"
        
        # 检查可选的环境变量
        for var in self.optional_env_vars:
            value = os.getenv(var)
            if value:
                env_status[var] = "✓ 已设置"
            else:
                env_status[var] = "○ 未设置 (可选)"
        
        # 显示状态
        for var, status in env_status.items():
            logger.info(f"{var}: {status}")
        
        # 检查是否有必要的变量未设置
        missing_required = [var for var in self.required_env_vars if not os.getenv(var)]
        if missing_required:
            logger.error(f"缺少必要的环境变量: {missing_required}")
            return env_status
        
        logger.info("环境变量检查通过")
        return env_status
    
    def check_data_files(self) -> bool:
        """检查数据文件"""
        logger.info("检查数据文件...")
        
        # 检查视频文件夹
        video_folder = Path("DADA-2000-videos")
        if not video_folder.exists():
            logger.error("视频文件夹不存在: DADA-2000-videos")
            return False
        
        # 检查目标视频文件
        target_videos = []
        for i in range(1, 6):  # images_1_* 到 images_5_*
            pattern = f"images_{i}_*.avi"
            videos = sorted(video_folder.glob(pattern))
            target_videos.extend(videos)
            if len(target_videos) >= 100:
                break
        
        target_videos = target_videos[:100]
        
        if len(target_videos) < 100:
            logger.warning(f"目标视频文件不足: 找到{len(target_videos)}个，需要100个")
        else:
            logger.info(f"找到{len(target_videos)}个目标视频文件")
        
        # 检查ground truth文件
        gt_file = Path("result/groundtruth_labels.csv")
        if not gt_file.exists():
            logger.error("Ground truth文件不存在: result/groundtruth_labels.csv")
            return False
        
        logger.info("数据文件检查通过")
        return True
    
    def test_azure_connection(self) -> bool:
        """测试Azure连接"""
        logger.info("测试Azure连接...")
        
        try:
            from azure.identity import DefaultAzureCredential
            from azure.ai.ml import MLClient
            
            # 获取Azure配置
            subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            resource_group = os.getenv("AZURE_RESOURCE_GROUP")
            workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
            
            if not all([subscription_id, resource_group, workspace_name]):
                logger.warning("Azure配置不完整，跳过连接测试")
                logger.info("请设置 AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME 环境变量")
                return True
            
            # 尝试连接
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
            
            # 测试连接
            workspace = ml_client.workspaces.get(workspace_name)
            logger.info(f"成功连接到Azure ML工作区: {workspace.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Azure连接测试失败: {e}")
            logger.info("请检查Azure凭据和配置")
            return False
    
    def test_openai_connection(self) -> bool:
        """测试OpenAI连接"""
        logger.info("测试OpenAI连接...")
        
        try:
            import openai
            from openai import AzureOpenAI
            
            # 获取配置
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment = os.getenv("VISION_DEPLOYMENT_NAME", "gpt-4.1")
            
            if not all([api_key, endpoint]):
                logger.error("OpenAI配置不完整")
                return False
            
            # 创建客户端
            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=endpoint
            )
            
            # 测试连接
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "测试连接"}],
                max_tokens=10
            )
            
            logger.info("OpenAI连接测试成功")
            return True
            
        except Exception as e:
            logger.error(f"OpenAI连接测试失败: {e}")
            return False
    
    def generate_env_template(self):
        """生成环境变量模板"""
        logger.info("生成环境变量模板...")
        
        template = """# Azure ML Ghost Probing Detection 环境变量配置
# 请填写以下变量的值，然后重命名为 .env 文件

# Azure OpenAI配置 (必需)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
VISION_DEPLOYMENT_NAME=gpt-4.1

# Azure Whisper配置 (必需)
AZURE_WHISPER_KEY=your_azure_whisper_key
AZURE_WHISPER_DEPLOYMENT=your_whisper_deployment
AZURE_WHISPER_ENDPOINT=https://your-whisper-endpoint.cognitiveservices.azure.com

# Azure ML配置 (可选，用于自动提交作业)
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_RESOURCE_GROUP=your_resource_group
AZURE_WORKSPACE_NAME=your_workspace_name
AZURE_COMPUTE_NAME=gpu-cluster-a100

# 其他配置
VISION_API_TYPE=Azure
AUDIO_API_TYPE=Azure
OPENAI_API_VERSION=2024-02-15-preview
"""
        
        with open(".env.template", "w", encoding="utf-8") as f:
            f.write(template)
        
        logger.info("环境变量模板已生成: .env.template")
        logger.info("请填写配置值后重命名为 .env 文件")
    
    def generate_submission_script(self):
        """生成提交脚本"""
        logger.info("生成提交脚本...")
        
        script = """#!/bin/bash
# Azure ML Ghost Probing Detection 提交脚本

# 设置错误退出
set -e

# 检查环境变量
if [ -z "$AZURE_SUBSCRIPTION_ID" ]; then
    echo "错误: 请设置 AZURE_SUBSCRIPTION_ID 环境变量"
    exit 1
fi

if [ -z "$AZURE_RESOURCE_GROUP" ]; then
    echo "错误: 请设置 AZURE_RESOURCE_GROUP 环境变量"
    exit 1
fi

if [ -z "$AZURE_WORKSPACE_NAME" ]; then
    echo "错误: 请设置 AZURE_WORKSPACE_NAME 环境变量"
    exit 1
fi

# 提交作业
echo "提交Ghost Probing Detection作业到Azure ML..."
python submit_azure_ghost_probing_job.py \\
    --subscription-id "$AZURE_SUBSCRIPTION_ID" \\
    --resource-group "$AZURE_RESOURCE_GROUP" \\
    --workspace-name "$AZURE_WORKSPACE_NAME" \\
    --compute-name "${AZURE_COMPUTE_NAME:-gpu-cluster-a100}"

echo "作业提交完成！"
"""
        
        with open("submit_ghost_probing_job.sh", "w", encoding="utf-8") as f:
            f.write(script)
        
        # 设置执行权限
        os.chmod("submit_ghost_probing_job.sh", 0o755)
        
        logger.info("提交脚本已生成: submit_ghost_probing_job.sh")
    
    def run_complete_check(self):
        """运行完整的检查"""
        logger.info("=== Azure ML Ghost Probing Detection 环境检查 ===")
        
        checks = [
            ("依赖项检查", self.check_dependencies),
            ("文件检查", self.check_files),
            ("数据文件检查", self.check_data_files),
        ]
        
        results = {}
        for name, check_func in checks:
            try:
                results[name] = check_func()
            except Exception as e:
                logger.error(f"{name}失败: {e}")
                results[name] = False
        
        # 环境变量检查
        env_status = self.check_environment_variables()
        results["环境变量检查"] = all("✗" not in status for status in env_status.values())
        
        # 可选测试
        if results["环境变量检查"]:
            results["OpenAI连接测试"] = self.test_openai_connection()
            results["Azure连接测试"] = self.test_azure_connection()
        
        # 总结
        logger.info("=== 检查结果总结 ===")
        for name, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            logger.info(f"{name}: {status}")
        
        all_passed = all(results.values())
        
        if all_passed:
            logger.info("🎉 所有检查通过！可以开始使用Azure ML进行鬼探头检测")
        else:
            logger.warning("⚠️ 部分检查未通过，请解决问题后再试")
        
        # 生成辅助文件
        self.generate_env_template()
        self.generate_submission_script()
        
        return all_passed


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Azure ML Ghost Probing Detection Setup')
    parser.add_argument('--generate-templates', action='store_true', help='仅生成模板文件')
    parser.add_argument('--test-connection', action='store_true', help='仅测试连接')
    
    args = parser.parse_args()
    
    setup = AzureMLGhostProbingSetup()
    
    if args.generate_templates:
        setup.generate_env_template()
        setup.generate_submission_script()
    elif args.test_connection:
        setup.test_openai_connection()
        setup.test_azure_connection()
    else:
        setup.run_complete_check()


if __name__ == "__main__":
    main()