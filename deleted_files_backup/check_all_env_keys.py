#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查所有可能的环境变量中的Gemini API Keys
"""

import os
import google.generativeai as genai

def test_api_key(api_key, key_name):
    """测试单个API Key"""
    if not api_key:
        return False
    
    print(f"🔑 测试 {key_name}: {api_key[:15]}...")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content(
            "请回复: OK",
            generation_config={"temperature": 0.1, "max_output_tokens": 10}
        )
        
        if response and response.text:
            print(f"✅ {key_name}: 可用")
            return True
        else:
            print(f"❌ {key_name}: 响应为空")
            return False
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"⚠️ {key_name}: 配额已用完")
        else:
            print(f"❌ {key_name}: 错误")
        return False

def main():
    print("🔍 检查所有环境变量中的API Keys")
    print("=" * 50)
    
    # 检查所有环境变量
    env_vars = os.environ
    gemini_keys = {}
    
    # 查找所有可能的gemini相关环境变量
    for key, value in env_vars.items():
        key_lower = key.lower()
        if ('gemini' in key_lower or 'google' in key_lower) and 'key' in key_lower:
            gemini_keys[key] = value
            print(f"📋 发现: {key} = {value[:15]}...")
    
    # 也检查一些常见的变体
    common_variants = [
        'GEMINI_API_KEY',
        'GEMINI_API_KEY_1', 
        'GEMINI_API_KEY_2',
        'GOOGLE_API_KEY',
        'GOOGLE_GEMINI_KEY',
        'GEMINI_KEY',
        'GEMINI_KEY_1',
        'GEMINI_KEY_2'
    ]
    
    for variant in common_variants:
        value = os.getenv(variant)
        if value and variant not in gemini_keys:
            gemini_keys[variant] = value
            print(f"📋 发现: {variant} = {value[:15]}...")
    
    if not gemini_keys:
        print("❌ 未找到任何Gemini API Key环境变量")
        return
    
    print(f"\n🧪 测试发现的 {len(gemini_keys)} 个API Keys:")
    working_keys = []
    
    for key_name, api_key in gemini_keys.items():
        if test_api_key(api_key, key_name):
            working_keys.append((key_name, api_key))
    
    print(f"\n📊 结果:")
    if working_keys:
        print(f"✅ 可用的API Keys: {len(working_keys)} 个")
        for key_name, api_key in working_keys:
            print(f"  - {key_name}: {api_key[:15]}...")
    else:
        print("❌ 没有可用的API Keys")

if __name__ == "__main__":
    main()