#!/usr/bin/env python3
"""
调试版本 - 查看Azure ML挂载的实际目录结构
"""

import os
import sys
from pathlib import Path

def debug_directory_structure():
    """调试目录结构"""
    print("=" * 60)
    print("🔍 Azure ML 挂载点调试信息")
    print("=" * 60)
    
    # 当前工作目录
    cwd = os.getcwd()
    print(f"📁 当前工作目录: {cwd}")
    
    # 列出当前目录的内容
    print(f"📂 当前目录内容:")
    try:
        for item in os.listdir('.'):
            item_path = Path(item)
            if item_path.is_dir():
                print(f"   📁 {item}/")
            else:
                print(f"   📄 {item}")
    except Exception as e:
        print(f"❌ 无法列出当前目录: {e}")
    
    # 查找所有可能的挂载点
    print(f"\n🔍 搜索可能的挂载点:")
    search_paths = [
        "/mnt",
        "/mnt/azureml", 
        cwd,
        os.path.join(cwd, "inputs"),
        os.path.join(cwd, "data"),
        "/tmp"
    ]
    
    for search_path in search_paths:
        try:
            if os.path.exists(search_path):
                print(f"✅ 路径存在: {search_path}")
                # 列出子目录
                items = os.listdir(search_path)
                for item in items[:5]:  # 只显示前5个
                    item_full = os.path.join(search_path, item)
                    if os.path.isdir(item_full):
                        print(f"   📁 {item}/")
                    else:
                        print(f"   📄 {item}")
                if len(items) > 5:
                    print(f"   ... 还有 {len(items) - 5} 个项目")
            else:
                print(f"❌ 路径不存在: {search_path}")
        except Exception as e:
            print(f"❌ 检查路径 {search_path} 时出错: {e}")
    
    # 递归搜索video相关目录
    print(f"\n🎬 递归搜索video相关目录:")
    try:
        # 从根目录和工作目录开始搜索
        for root_dir in [cwd, "/mnt"]:
            if os.path.exists(root_dir):
                for root, dirs, files in os.walk(root_dir):
                    # 限制搜索深度
                    if root.count(os.sep) - root_dir.count(os.sep) > 5:
                        continue
                    
                    if any(keyword in root.lower() for keyword in ['video', 'data', 'input']):
                        print(f"📁 找到相关目录: {root}")
                        # 检查是否有.avi文件
                        avi_files = [f for f in files if f.endswith('.avi')]
                        if avi_files:
                            print(f"   🎬 找到 {len(avi_files)} 个.avi文件")
                            for avi in avi_files[:3]:
                                print(f"      • {avi}")
                            if len(avi_files) > 3:
                                print(f"      ... 还有 {len(avi_files) - 3} 个文件")
                        else:
                            print(f"   ⚠️  目录中没有.avi文件")
    except Exception as e:
        print(f"❌ 递归搜索时出错: {e}")
    
    # 检查环境变量
    print(f"\n🔧 相关环境变量:")
    env_vars = ['AZUREML_DATAREFERENCE_video_data', 'AZUREML_DATA_INPUT_video_data', 'PWD', 'HOME']
    for var in env_vars:
        value = os.environ.get(var, "未设置")
        print(f"   {var} = {value}")
    
    print("=" * 60)

if __name__ == "__main__":
    debug_directory_structure()