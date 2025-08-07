#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试可用的Gemini API Keys
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_api_key(api_key, key_name):
    """测试单个API Key"""
    if not api_key:
        print(f"❌ {key_name}: 未设置")
        return False
    
    print(f"🔑 测试 {key_name}: {api_key[:15]}...")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # 简单测试
        response = model.generate_content(
            "请回复: OK",
            generation_config={"temperature": 0.1, "max_output_tokens": 10}
        )
        
        if response and response.text:
            print(f"✅ {key_name}: 可用 - 响应: {response.text.strip()}")
            return True
        else:
            print(f"❌ {key_name}: 响应为空")
            return False
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"⚠️ {key_name}: 配额已用完")
        elif "403" in error_msg or "permission" in error_msg.lower():
            print(f"❌ {key_name}: 权限错误")
        else:
            print(f"❌ {key_name}: 错误 - {error_msg}")
        return False

def main():
    print("🔍 测试所有可用的Gemini API Keys")
    print("=" * 50)
    
    load_dotenv()
    
    # 测试所有可能的API Key
    keys_to_test = [
        ("GEMINI_API_KEY", os.getenv('GEMINI_API_KEY')),
        ("GEMINI_API_KEY_2", os.getenv('GEMINI_API_KEY_2')),
        ("GOOGLE_API_KEY", os.getenv('GOOGLE_API_KEY')),
        ("GEMINI_KEY", os.getenv('GEMINI_KEY')),
    ]
    
    working_keys = []
    
    for key_name, api_key in keys_to_test:
        if test_api_key(api_key, key_name):
            working_keys.append((key_name, api_key))
        print()
    
    print("📊 测试结果总结:")
    if working_keys:
        print(f"✅ 可用的API Keys: {len(working_keys)} 个")
        for key_name, _ in working_keys:
            print(f"  - {key_name}")
        
        # 如果有可用的key，我们可以继续处理
        print(f"\n🚀 发现可用API Key，可以继续处理剩余视频！")
        return working_keys[0]  # 返回第一个可用的key
    else:
        print("❌ 没有可用的API Keys")
        print("💡 建议:")
        print("  1. 检查.env文件中的API Key配置")
        print("  2. 确认API Key有效且有剩余配额")
        print("  3. 等待配额重置（通常在UTC时间00:00）")
        return None

if __name__ == "__main__":
    result = main()