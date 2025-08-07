#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
创建.env文件并设置第二个Gemini API Key
"""

import os
import shutil

def create_env_file():
    """创建.env文件"""
    print("🔧 创建.env文件设置助手")
    print("=" * 50)
    
    # 复制.envsample到.env
    if os.path.exists('.envsample'):
        shutil.copy('.envsample', '.env')
        print("✅ 已从.envsample创建.env文件")
    else:
        # 创建基本的.env文件
        with open('.env', 'w') as f:
            f.write("# Gemini API Keys\n")
            f.write("GEMINI_API_KEY=\n")
            f.write("GEMINI_API_KEY_2=\n")
            f.write("GEMINI_MODEL=gemini-2.0-flash\n")
        print("✅ 已创建基本.env文件")
    
    print("\n📝 请按以下步骤操作:")
    print("1. 编辑.env文件")
    print("2. 将你的两个Gemini API Key分别设置为:")
    print("   GEMINI_API_KEY=你的第一个key")
    print("   GEMINI_API_KEY_2=你的第二个key")
    print("3. 保存文件")
    print("4. 重新运行处理脚本")
    
    print(f"\n💡 当前第一个key已在环境变量中: {os.getenv('GEMINI_API_KEY', 'NONE')[:15]}...")
    print("🔑 你只需要在.env文件中添加第二个key")
    
    # 显示.env文件当前内容
    if os.path.exists('.env'):
        print(f"\n📄 当前.env文件内容:")
        with open('.env', 'r') as f:
            content = f.read()
            print(content)

def main():
    create_env_file()
    
    print(f"\n🚀 设置完成后，运行以下命令继续处理:")
    print(f"python gemini_final_push_200rpd.py")

if __name__ == "__main__":
    main()