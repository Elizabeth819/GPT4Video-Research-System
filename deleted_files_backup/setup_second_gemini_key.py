#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设置第二个Gemini API Key并完成剩余视频处理
"""

import os
import tempfile
import google.generativeai as genai

def setup_second_key_from_input():
    """从用户输入设置第二个API Key"""
    print("🔑 根据您之前提到的，您在.env文件中有两个gemini api key")
    print("💡 当前只检测到一个API Key，可能第二个key的环境变量名不同")
    print("\n请选择操作方式:")
    print("1. 手动输入第二个API Key")
    print("2. 告诉我第二个API Key的环境变量名")
    print("3. 等待明天配额重置")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 手动输入API Key
        import getpass
        api_key = getpass.getpass("请输入第二个Gemini API Key: ").strip()
        if api_key:
            return test_and_use_key(api_key, "手动输入的Key")
    
    elif choice == "2":
        # 询问环境变量名
        var_name = input("请输入第二个API Key的环境变量名: ").strip()
        if var_name:
            api_key = os.getenv(var_name)
            if api_key:
                return test_and_use_key(api_key, var_name)
            else:
                print(f"❌ 环境变量 {var_name} 未设置或为空")
    
    elif choice == "3":
        print("⏰ 好的，明天配额重置后再继续处理")
        print("💡 建议使用命令: python gemini_final_push_200rpd.py")
        return None
    
    else:
        print("❌ 无效选择")
        return None

def test_and_use_key(api_key, key_name):
    """测试API Key并返回可用的模型"""
    print(f"🔑 测试 {key_name}: {api_key[:15]}...")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content(
            "Test - return OK",
            generation_config={"temperature": 0.1, "max_output_tokens": 10}
        )
        
        if response and response.text:
            print(f"✅ {key_name} 可用！")
            return model
        else:
            print(f"❌ {key_name} 响应为空")
            return None
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"⚠️ {key_name} 配额已用完")
        else:
            print(f"❌ {key_name} 错误: {error_msg}")
        return None

def continue_with_processing(model):
    """使用可用的模型继续处理"""
    print("\n🚀 发现可用API Key，开始处理剩余视频...")
    
    # 导入处理模块
    import sys
    import importlib.util
    
    # 动态导入处理脚本的函数
    spec = importlib.util.spec_from_file_location("processor", "gemini_final_push_200rpd.py")
    processor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(processor)
    
    # 获取剩余视频
    remaining_videos = processor.get_remaining_videos()
    print(f"📋 剩余未处理视频: {len(remaining_videos)} 个")
    
    if len(remaining_videos) == 0:
        print("🎉 所有视频已处理完成！")
        return
    
    # 开始处理
    output_dir = "result/gemini-balanced-full"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failed_videos = []
    
    from tqdm import tqdm
    import time
    
    for i, video_path in enumerate(tqdm(remaining_videos, desc="处理视频"), 1):
        print(f"\n[{i}/{len(remaining_videos)}] 处理: {os.path.basename(video_path)}")
        
        if processor.process_single_video(video_path, model, output_dir):
            success_count += 1
        else:
            failed_videos.append(video_path)
        
        if i % 10 == 0:
            print(f"📊 进度: {success_count}/{i} 成功")
        
        time.sleep(1)  # 避免API限制
    
    # 统计结果
    total_processed = 49 + success_count
    print(f"\n🎯 处理完成!")
    print(f"  📊 本次成功: {success_count}/{len(remaining_videos)}")
    print(f"  📊 总体进度: {total_processed}/99 ({total_processed/99*100:.1f}%)")
    
    if total_processed >= 99:
        print("🎉 恭喜！已完成全部99个视频的Gemini处理！")

def main():
    print("🔧 Gemini第二API Key设置助手")
    print("=" * 50)
    
    # 检查当前状态
    from gemini_final_push_200rpd import get_remaining_videos
    remaining = get_remaining_videos()
    print(f"📋 当前剩余: {len(remaining)} 个视频需要处理")
    
    if len(remaining) == 0:
        print("🎉 所有视频已完成！")
        return
    
    # 尝试设置第二个API Key
    model = setup_second_key_from_input()
    
    if model:
        # 继续处理
        continue_with_processing(model)
    else:
        print("\n💡 其他选项:")
        print("  1. 等待明天配额重置 (UTC 00:00)")
        print("  2. 获取新的Gemini API Key")
        print("  3. 使用其他Google账户创建API Key")

if __name__ == "__main__":
    main()