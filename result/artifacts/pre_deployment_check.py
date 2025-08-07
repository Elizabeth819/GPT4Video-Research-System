#!/usr/bin/env python3
"""
部署前检查脚本
确保所有必要的组件都准备就绪
"""

import os
import sys
from pathlib import Path
import subprocess

def check_environment_variables():
    """检查环境变量"""
    print("🔍 检查环境变量...")
    
    required_vars = [
        'AZURE_SUBSCRIPTION_ID',
        'AZURE_RESOURCE_GROUP', 
        'AZURE_WORKSPACE_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ 缺少以下环境变量:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    
    print("✅ 环境变量检查通过")
    return True

def check_files():
    """检查必要文件"""
    print("\n📁 检查必要文件...")
    
    required_files = [
        'create_videochat2_a100_cluster.yml',
        'videochat2_ghost_probing_job.yml',
        'videochat2_environment.yml',
        'deploy_videochat2_cluster.py',
        'quick_start_videochat2_gpu.sh',
        'videochat2_ghost_detection/videochat2_ghost_detection.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少以下文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ 必要文件检查通过")
    return True

def check_video_data():
    """检查视频数据"""
    print("\n🎬 检查视频数据...")
    
    video_dir = Path("./DADA-2000-videos")
    if not video_dir.exists():
        print(f"❌ 视频目录不存在: {video_dir}")
        return False
    
    # 统计目标视频
    video_count = 0
    for i in range(1, 6):  # 1 to 5
        pattern = f"images_{i}_*.avi"
        videos = list(video_dir.glob(pattern))
        video_count += len(videos)
    
    if video_count == 0:
        print("❌ 没有找到目标视频文件")
        return False
    
    print(f"✅ 找到 {video_count} 个视频文件")
    print(f"   - 将处理前 {min(100, video_count)} 个视频")
    return True

def check_azure_cli():
    """检查Azure CLI"""
    print("\n🔧 检查Azure CLI...")
    
    try:
        result = subprocess.run(['az', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Azure CLI 可用")
            return True
        else:
            print("❌ Azure CLI 不可用")
            return False
    except FileNotFoundError:
        print("❌ Azure CLI 未安装")
        return False

def check_python_packages():
    """检查Python包"""
    print("\n🐍 检查Python包...")
    
    required_packages = [
        ('azure-ai-ml', 'azure.ai.ml'),
        ('azure-identity', 'azure.identity'),
        ('torch', 'torch'),
        ('transformers', 'transformers')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ 缺少以下Python包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n安装命令:")
        print("pip install " + " ".join(missing_packages))
        return False
    
    print("✅ Python包检查通过")
    return True

def check_azure_login():
    """检查Azure登录状态"""
    print("\n🔐 检查Azure登录状态...")
    
    try:
        result = subprocess.run(['az', 'account', 'show'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Azure 已登录")
            return True
        else:
            print("❌ Azure 未登录")
            print("请运行: az login")
            return False
    except Exception as e:
        print(f"❌ 检查Azure登录状态失败: {e}")
        return False

def check_model_availability():
    """检查模型可用性"""
    print("\n🤖 检查模型可用性...")
    
    model_dir = Path("./models/videochat2-hd")
    if model_dir.exists():
        print("✅ 本地模型目录存在")
        return True
    else:
        print("⚠️  本地模型目录不存在")
        print("   系统将从HuggingFace自动下载模型")
        print("   请确保已获得模型访问权限")
        return True

def main():
    """主检查函数"""
    print("🚀 VideoChat2 A100 部署前检查")
    print("=" * 50)
    
    checks = [
        check_environment_variables,
        check_files,
        check_video_data,
        check_azure_cli,
        check_python_packages,
        check_azure_login,
        check_model_availability
    ]
    
    results = []
    for check in checks:
        result = check()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📋 检查结果汇总:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ 通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有检查都通过！可以开始部署。")
        print("\n🚀 运行部署命令:")
        print("   ./quick_start_videochat2_gpu.sh deploy")
        return True
    else:
        failed = total - passed
        print(f"❌ 失败: {failed}/{total}")
        print("\n⚠️  请解决上述问题后再次运行检查。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)