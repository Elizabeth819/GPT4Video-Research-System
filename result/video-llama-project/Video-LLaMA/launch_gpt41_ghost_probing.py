#!/usr/bin/env python3
"""
启动GPT-4.1平衡版鬼探头检测作业
从Video-LLaMA目录启动Azure ML作业
"""

import os
import sys
import subprocess
from pathlib import Path

# 设置Azure ML环境变量
os.environ["AZURE_SUBSCRIPTION_ID"] = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a"
os.environ["AZURE_RESOURCE_GROUP"] = "video-llama2-ghost-probing-rg"
os.environ["AZURE_WORKSPACE_NAME"] = "video-llama2-ghost-probing-ws"

# 切换到父目录
parent_dir = Path(__file__).parent.parent
print(f"切换到工作目录: {parent_dir}")
os.chdir(parent_dir)

def run_environment_test():
    """运行环境测试"""
    print("🧪 运行环境测试...")
    try:
        result = subprocess.run([
            sys.executable, "test_azure_setup.py"
        ], capture_output=True, text=True, cwd=parent_dir)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return False

def submit_job():
    """提交作业"""
    print("🚀 提交GPT-4.1平衡版鬼探头检测作业...")
    try:
        result = subprocess.run([
            sys.executable, "submit_gpt41_balanced_job.py", "--check-only"
        ], capture_output=True, text=True, cwd=parent_dir)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ 环境检查通过，现在提交真实作业...")
            result = subprocess.run([
                sys.executable, "submit_gpt41_balanced_job.py"
            ], cwd=parent_dir)
            return result.returncode == 0
        else:
            print("❌ 环境检查失败")
            return False
            
    except Exception as e:
        print(f"❌ 作业提交失败: {e}")
        return False

def show_status():
    """显示当前状态"""
    print("=" * 60)
    print("🎯 GPT-4.1 Balanced Ghost Probing Detection")
    print("=" * 60)
    print(f"Azure 订阅ID: {os.environ['AZURE_SUBSCRIPTION_ID']}")
    print(f"资源组: {os.environ['AZURE_RESOURCE_GROUP']}")
    print(f"工作区: {os.environ['AZURE_WORKSPACE_NAME']}")
    print(f"工作目录: {parent_dir}")
    print("=" * 60)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch GPT-4.1 Ghost Probing Job')
    parser.add_argument('--test-only', action='store_true', help='仅运行环境测试')
    parser.add_argument('--submit', action='store_true', help='直接提交作业')
    
    args = parser.parse_args()
    
    show_status()
    
    if args.test_only:
        success = run_environment_test()
        if success:
            print("✅ 环境测试通过")
        else:
            print("❌ 环境测试失败")
    elif args.submit:
        success = submit_job()
        if success:
            print("✅ 作业提交成功")
        else:
            print("❌ 作业提交失败")
    else:
        print("请选择操作:")
        print("  --test-only: 仅测试环境")
        print("  --submit: 提交作业")
        print("\n或直接运行以下命令:")
        print(f"cd {parent_dir}")
        print("python test_azure_setup.py")
        print("python submit_gpt41_balanced_job.py")

if __name__ == "__main__":
    main()