#!/usr/bin/env python
"""
在Azure ML上运行DriveMM推理的简化脚本
"""

import os
import json
import subprocess
import sys

def run_drivemm_inference():
    """运行DriveMM推理的主函数"""
    
    print("🚀 开始Azure ML DriveMM推理")
    print("=" * 60)
    
    # 检查配置文件
    if not os.path.exists('config.json'):
        print("❌ 找不到config.json文件")
        print("📝 请复制 config.json.example 为 config.json 并填入实际配置")
        return False
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print(f"📋 配置信息:")
    print(f"  订阅ID: {config.get('subscription_id', 'N/A')}")
    print(f"  资源组: {config.get('resource_group', 'N/A')}")
    print(f"  工作区: {config.get('workspace_name', 'N/A')}")
    print(f"  计算目标: {config.get('compute_target', 'drivemm-a100-cluster')}")
    
    # 方法1: 使用Azure CLI提交作业
    print("\n📝 方法1: 使用Azure CLI提交作业")
    print("=" * 40)
    
    print("1. 设置Azure存储连接字符串:")
    print("   export AZURE_STORAGE_CONNECTION_STRING='你的存储连接字符串'")
    
    print("\n2. 登录Azure:")
    print("   az login")
    
    print("\n3. 设置默认订阅:")
    print(f"   az account set --subscription {config.get('subscription_id', '<订阅ID>')}")
    
    print("\n4. 提交作业:")
    print(f"   az ml job create --file azure_ml_drivemm_real_job.yml --workspace-name {config.get('workspace_name', '<工作区>')} --resource-group {config.get('resource_group', '<资源组>')}")
    
    # 方法2: 使用Python SDK
    print("\n📝 方法2: 使用Python SDK")
    print("=" * 40)
    
    print("1. 安装依赖:")
    print("   pip install azure-ai-ml azure-identity")
    
    print("\n2. 运行设置脚本:")
    print("   python setup_drivemm_azure.py")
    
    # 方法3: 直接在compute instance上运行
    print("\n📝 方法3: 在Compute Instance上直接运行")
    print("=" * 40)
    
    print("1. 创建或启动compute instance")
    print("2. 在terminal中运行:")
    print("   git clone <your-repo>")
    print("   cd GPT4Video-cobra-auto")
    print("   conda env create -f azure_drivemm_environment.yml")
    print("   conda activate drivemm_inference")
    print("   python azure_drivemm_real_inference.py")
    
    # 检查文件是否存在
    print("\n📁 检查必要文件:")
    required_files = [
        'azure_drivemm_real_inference.py',
        'azure_ml_drivemm_real_job.yml',
        'azure_drivemm_environment.yml'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ⚠️  {file} (缺失)")
    
    return True

def show_monitoring_info():
    """显示监控信息"""
    print("\n📊 监控和调试")
    print("=" * 40)
    
    print("1. 查看作业状态:")
    print("   az ml job show --name <job-name> --workspace-name <workspace> --resource-group <rg>")
    
    print("\n2. 查看作业日志:")
    print("   az ml job logs --name <job-name> --workspace-name <workspace> --resource-group <rg>")
    
    print("\n3. 在Azure ML Studio中监控:")
    print("   https://ml.azure.com")
    
    print("\n4. 常见问题:")
    print("   - GPU配额不足: 需要申请A100配额")
    print("   - 模型下载失败: 检查网络连接和HuggingFace权限")
    print("   - 内存不足: 增加shm_size或使用更大的VM")

def show_gpu_requirements():
    """显示GPU要求和建议"""
    print("\n🔧 DriveMM模型GPU要求分析")
    print("=" * 60)
    
    print("📊 DriveMM模型规格:")
    print("  - 模型名称: DriveMM/DriveMM")
    print("  - 参数量: 8.45B (84.5亿参数)")
    print("  - 模型大小: ~17GB (下载大小)")
    print("  - 精度: bfloat16")
    print("  - 推理框架: HuggingFace Transformers")
    
    print("\n💾 GPU内存要求:")
    print("  - 模型加载: ~17GB VRAM (bfloat16)")
    print("  - 推理缓存: ~3-5GB VRAM")
    print("  - 系统开销: ~2-3GB VRAM")
    print("  - 总计需要: ~22-25GB VRAM")
    
    print("\n🎯 Azure GPU选择建议:")
    print("  ✅ 推荐选择: A100 (40GB/80GB)")
    print("     - Standard_NC24ads_A100_v4 (1x A100 40GB) - 足够运行")
    print("     - Standard_NC48ads_A100_v4 (2x A100 40GB) - 更快推理")
    print("     - Standard_NC96ads_A100_v4 (4x A100 40GB) - 最佳性能")
    
    print("\n  ⚠️  替代选择: H100 (如果有配额)")
    print("     - Standard_ND96isr_H100_v5 (8x H100 80GB) - 最高性能")
    
    print("\n  ❌ 不推荐:")
    print("     - V100 (16GB) - 内存不足")
    print("     - RTX 6000 (24GB) - 勉强够用但性能较差")
    
    print("\n🏆 最佳配置推荐:")
    print("  - 生产环境: Standard_NC96ads_A100_v4 (4x A100 40GB)")
    print("  - 开发测试: Standard_NC24ads_A100_v4 (1x A100 40GB) ⭐推荐")

def check_gpu_quota():
    """检查GPU配额"""
    print("\n📋 检查GPU配额:")
    print("  1. 登录Azure Portal")
    print("  2. 进入订阅 -> 使用量 + 配额")
    print("  3. 搜索 'NC24ads A100' 或 'NC96ads A100'")
    print("  4. 检查当前配额和使用情况")
    print("  5. 如需增加配额，点击 '请求增加配额'")

def show_cost_estimate():
    """显示成本估算"""
    print("\n💰 成本估算 (美国东部地区):")
    print("  - Standard_NC24ads_A100_v4: ~$3.67/小时 (1x A100)")
    print("  - Standard_NC48ads_A100_v4: ~$7.35/小时 (2x A100)")
    print("  - Standard_NC96ads_A100_v4: ~$14.69/小时 (4x A100)")
    print("  - 预计推理时间: 2-4小时 (处理dada-videos中的视频)")
    print("  - 预计总成本: $7-60 (取决于选择的VM)")

def main():
    """主函数"""
    print("🔧 Azure ML DriveMM推理设置向导")
    print("=" * 60)
    
    # 显示GPU要求
    show_gpu_requirements()
    check_gpu_quota()
    show_cost_estimate()
    
    print("\n" + "=" * 60)
    
    # 运行主要设置
    success = run_drivemm_inference()
    
    if success:
        # 显示监控信息
        show_monitoring_info()
        
        print("\n🎯 下一步:")
        print("1. 确认GPU配额足够 (推荐A100)")
        print("2. 选择上面的方法之一运行DriveMM推理")
        print("3. 在Azure ML Studio中监控作业进度")
        print("4. 下载结果进行分析")
        print("5. 与其他模型结果对比")
    
    return success

if __name__ == "__main__":
    main()
