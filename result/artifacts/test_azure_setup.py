#!/usr/bin/env python3
"""
测试Azure ML环境设置
验证您的Azure ML工作区连接和GPT-4.1配置
"""

import os
import sys
import logging
from datetime import datetime

# 设置Azure ML环境变量
os.environ["AZURE_SUBSCRIPTION_ID"] = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a"
os.environ["AZURE_RESOURCE_GROUP"] = "video-llama2-ghost-probing-rg"
os.environ["AZURE_WORKSPACE_NAME"] = "video-llama2-ghost-probing-ws"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_azure_ml_connection():
    """测试Azure ML连接"""
    logger.info("🔍 测试Azure ML连接...")
    
    try:
        from azure.identity import DefaultAzureCredential
        from azure.ai.ml import MLClient
        
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id="0d3f39ba-7349-4bd7-8122-649ff18f0a4a",
            resource_group_name="video-llama2-ghost-probing-rg",
            workspace_name="video-llama2-ghost-probing-ws"
        )
        
        # 测试连接
        workspace = ml_client.workspaces.get("video-llama2-ghost-probing-ws")
        logger.info(f"✅ 成功连接到工作区: {workspace.name}")
        logger.info(f"   位置: {workspace.location}")
        logger.info(f"   资源组: {workspace.resource_group}")
        
        # 列出计算资源
        logger.info("📊 可用的计算资源:")
        computes = ml_client.compute.list()
        for compute in computes:
            logger.info(f"   - {compute.name}: {compute.type} ({compute.provisioning_state})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Azure ML连接失败: {e}")
        return False

def test_openai_configuration():
    """测试OpenAI配置"""
    logger.info("🔍 检查OpenAI环境变量...")
    
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_WHISPER_KEY",
        "AZURE_WHISPER_ENDPOINT"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            logger.error(f"❌ 缺少环境变量: {var}")
        else:
            logger.info(f"✅ {var}: 已设置")
    
    if missing_vars:
        logger.error("❌ 请设置缺少的环境变量")
        logger.info("建议在 .env 文件中设置这些变量")
        return False
    
    # 测试OpenAI连接
    logger.info("🔍 测试OpenAI连接...")
    try:
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # 简单测试
        response = client.chat.completions.create(
            model=os.getenv("VISION_DEPLOYMENT_NAME", "gpt-4.1"),
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        logger.info("✅ OpenAI连接测试成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenAI连接测试失败: {e}")
        return False

def test_local_files():
    """测试本地文件"""
    logger.info("🔍 检查本地文件...")
    
    required_files = [
        "batch_ghost_probing_gpt41_balanced.py",
        "azure_ghost_probing_env.yml",
        "result/groundtruth_labels.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            logger.info(f"✅ {file_path}: 存在")
        else:
            logger.error(f"❌ {file_path}: 不存在")
            missing_files.append(file_path)
    
    # 检查视频文件
    video_folder = "DADA-2000-videos"
    if os.path.exists(video_folder):
        import glob
        target_videos = []
        for i in range(1, 6):
            pattern = f"{video_folder}/images_{i}_*.avi"
            videos = sorted(glob.glob(pattern))
            target_videos.extend(videos)
            if len(target_videos) >= 100:
                break
        
        target_videos = target_videos[:100]
        logger.info(f"✅ 找到 {len(target_videos)} 个目标视频文件")
        
        if len(target_videos) >= 100:
            logger.info("✅ 视频文件数量满足要求 (100个)")
        else:
            logger.warning(f"⚠️ 视频文件数量不足: {len(target_videos)}/100")
    else:
        logger.error(f"❌ 视频文件夹不存在: {video_folder}")
        missing_files.append(video_folder)
    
    return len(missing_files) == 0

def test_dependencies():
    """测试Python依赖"""
    logger.info("🔍 检查Python依赖...")
    
    required_packages = [
        "azure.ai.ml",
        "azure.identity",
        "openai",
        "pandas",
        "numpy",
        "cv2",
        "moviepy",
        "tqdm"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✅ {package}: 已安装")
        except ImportError:
            logger.error(f"❌ {package}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("❌ 请安装缺少的包:")
        logger.error(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def generate_test_report():
    """生成测试报告"""
    logger.info("=" * 60)
    logger.info("🧪 Azure ML GPT-4.1 Ghost Probing 环境测试")
    logger.info("=" * 60)
    
    tests = [
        ("Python依赖检查", test_dependencies),
        ("本地文件检查", test_local_files),
        ("OpenAI配置检查", test_openai_configuration),
        ("Azure ML连接检查", test_azure_ml_connection)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n📋 {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} 失败: {e}")
            results[test_name] = False
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("📊 测试结果总结")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 所有测试通过！可以提交Azure ML作业")
        logger.info("\n下一步:")
        logger.info("1. 设置必要的环境变量 (如果还未设置)")
        logger.info("2. 运行: python submit_gpt41_balanced_job.py")
    else:
        logger.error("❌ 部分测试失败，请解决问题后重试")
        logger.info("\n建议:")
        logger.info("1. 检查Azure凭据: az login")
        logger.info("2. 设置环境变量: source .env")
        logger.info("3. 安装缺少的依赖包")
    
    return all_passed

if __name__ == "__main__":
    generate_test_report()