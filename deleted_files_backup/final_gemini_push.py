#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最后冲刺 - 尝试完成剩余50个视频
先用API Key 2，如果不行就等明天
"""

import os
import json
import cv2
from moviepy.editor import VideoFileClip
import google.generativeai as genai
from dotenv import load_dotenv
import time
import pandas as pd
from tqdm import tqdm

def try_api_key_2():
    """尝试使用API Key 2"""
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY_2')
    
    print(f"🔑 尝试API Key 2: {api_key[:10]}...")
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # 测试API调用
        test_response = model.generate_content(
            "Test message - return just 'OK'",
            generation_config={"temperature": 0.1, "max_output_tokens": 10}
        )
        
        if test_response.text:
            print("✅ API Key 2 可用！")
            return model
        else:
            print("❌ API Key 2 响应为空")
            return None
            
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            print("⚠️ API Key 2 配额已用完")
        else:
            print(f"❌ API Key 2 错误: {str(e)}")
        return None

def get_remaining_videos():
    """获取剩余未处理的视频"""
    df = pd.read_csv('result/groundtruth_labels.csv', sep='\t')
    all_videos = df['video_id'].str.replace('.avi', '').tolist()
    
    output_dir = "result/gemini-balanced-full"
    remaining = []
    
    for vid in all_videos:
        if vid and isinstance(vid, str):
            video_path = f"DADA-2000-videos/{vid}.avi"
            video_id_formatted = vid.replace('images_', 'dada_')
            output_file = os.path.join(output_dir, f"actionSummary_{video_id_formatted}.json")
            
            if os.path.exists(video_path) and not os.path.exists(output_file):
                remaining.append(video_path)
    
    return remaining

def main():
    print("🚀 最后冲刺 - 完成剩余视频处理")
    print("=" * 50)
    
    # 检查剩余视频
    remaining_videos = get_remaining_videos()
    print(f"📋 剩余未处理视频: {len(remaining_videos)} 个")
    
    if len(remaining_videos) == 0:
        print("🎉 所有视频已处理完成！")
        return
    
    # 尝试API Key 2
    model = try_api_key_2()
    
    if model:
        print(f"🎯 使用API Key 2继续处理 {len(remaining_videos)} 个视频")
        print("📋 将创建新的处理脚本...")
        
        # 这里可以继续处理逻辑
        print("💡 API Key 2可用，建议运行完整处理脚本")
        
    else:
        print("❌ 两个API Key都已用完")
        print(f"📊 当前进度: {99 - len(remaining_videos)}/99 ({((99 - len(remaining_videos))/99*100):.1f}%)")
        print("⏰ 建议明天继续处理剩余视频")
        
        # 创建明天的处理计划
        with open("tomorrow_processing_plan.txt", "w", encoding='utf-8') as f:
            f.write("Gemini Processing Plan - Next Day\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Remaining videos: {len(remaining_videos)}\n")
            f.write(f"Current progress: {99 - len(remaining_videos)}/99\n\n")
            f.write("Command to run:\n")
            f.write("python gemini_continue_with_key1.py\n")
            f.write("or\n")
            f.write("python gemini_daily_batch.py\n")
        
        print("📝 已创建明天处理计划: tomorrow_processing_plan.txt")

if __name__ == "__main__":
    main()