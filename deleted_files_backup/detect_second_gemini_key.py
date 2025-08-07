#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动检测第二个Gemini API Key
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_api_key(api_key, key_name):
    """测试API Key"""
    if not api_key:
        return False
    
    print(f"🔑 测试 {key_name}: {api_key[:15]}...")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content(
            "Test",
            generation_config={"temperature": 0.1, "max_output_tokens": 5}
        )
        
        if response and response.text:
            print(f"✅ {key_name}: 可用！")
            return model
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
    print("🔍 自动检测第二个Gemini API Key")
    print("=" * 50)
    
    # 加载.env文件
    load_dotenv()
    
    # 尝试各种可能的环境变量名
    possible_keys = [
        'GEMINI_API_KEY',
        'GEMINI_API_KEY_1', 
        'GEMINI_API_KEY_2',
        'GEMINI_KEY_1',
        'GEMINI_KEY_2', 
        'GOOGLE_API_KEY',
        'GOOGLE_API_KEY_1',
        'GOOGLE_API_KEY_2',
        'GEMINI_KEY',
        'GOOGLE_GEMINI_KEY',
        'API_KEY_GEMINI',
        'API_KEY_GEMINI_2'
    ]
    
    working_model = None
    tested_keys = set()
    
    for key_name in possible_keys:
        api_key = os.getenv(key_name)
        if api_key and api_key not in tested_keys:
            tested_keys.add(api_key)
            result = test_api_key(api_key, key_name)
            if result:
                working_model = result
                print(f"🎉 找到可用的API Key: {key_name}")
                break
    
    if working_model:
        print(f"\n✅ 发现可用API Key，可以继续处理剩余视频！")
        
        # 直接启动处理
        try:
            print("🚀 开始处理剩余50个视频...")
            
            # 导入并运行处理逻辑
            from gemini_final_push_200rpd import get_remaining_videos, process_single_video
            from tqdm import tqdm
            import time
            
            remaining_videos = get_remaining_videos()
            print(f"📋 剩余视频: {len(remaining_videos)} 个")
            
            output_dir = "result/gemini-balanced-full"
            os.makedirs(output_dir, exist_ok=True)
            
            success_count = 0
            failed_videos = []
            
            for i, video_path in enumerate(tqdm(remaining_videos, desc="处理视频"), 1):
                print(f"\n[{i}/{len(remaining_videos)}] {os.path.basename(video_path)}")
                
                if process_single_video(video_path, working_model, output_dir):
                    success_count += 1
                else:
                    failed_videos.append(video_path)
                
                if i % 10 == 0:
                    print(f"📊 进度: {success_count}/{i}")
                
                time.sleep(1)
            
            total_processed = 49 + success_count
            print(f"\n🎯 处理完成!")
            print(f"  📊 成功: {success_count}/{len(remaining_videos)}")
            print(f"  📊 总进度: {total_processed}/99 ({total_processed/99*100:.1f}%)")
            
            if total_processed >= 99:
                print("🎉 全部99个视频处理完成！")
            
        except Exception as e:
            print(f"❌ 处理过程中出错: {e}")
    
    else:
        print(f"\n❌ 未找到可用的API Key")
        print(f"📊 已测试的唯一key数量: {len(tested_keys)}")
        
        # 检查.env文件内容
        env_file = ".env"
        if os.path.exists(env_file):
            print(f"\n🔍 检查 {env_file} 文件内容:")
            with open(env_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'gemini' in line.lower() or 'google' in line.lower():
                        # 隐藏实际的key值
                        if '=' in line:
                            key_part = line.split('=')[0]
                            print(f"  发现: {key_part}=...")
        else:
            print(f"⚠️ {env_file} 文件不存在")
            
        print(f"\n💡 建议:")
        print(f"  1. 检查.env文件中是否有第二个API Key")
        print(f"  2. 确认第二个key的环境变量名称")
        print(f"  3. 或等待明天配额重置（UTC 00:00）")

if __name__ == "__main__":
    main()