#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
手动输入Gemini API Key完成剩余50个视频
"""

import os
import json
import cv2
from moviepy.editor import VideoFileClip
import google.generativeai as genai
import time
import pandas as pd
from tqdm import tqdm
import shutil
import getpass

def get_balanced_prompt():
    """获取平衡版prompt - 与GPT-4.1完全相同"""
    return """You are an expert AI system analyzing sequential video frames from a moving vehicle's dashboard camera. Your task is to detect and analyze "ghost probing" (鬼探头) behavior - when objects (vehicles, pedestrians, cyclists, etc.) suddenly appear from concealed positions and potentially create collision risks.

**LAYERED DETECTION STRATEGY:**

**1. HIGH-CONFIDENCE Ghost Probing (use "ghost probing" in key_actions)**:
- Object appears EXTREMELY close (within 1-2 vehicle lengths, <3 meters)
- Object was completely hidden/concealed and suddenly becomes visible
- Creates IMMEDIATE collision risk requiring emergency response
- Appearance is sudden and unexpected from the ego vehicle's perspective

**2. POTENTIAL Ghost Probing (use "potential ghost probing" in key_actions)**:
- Object appears from partially concealed position
- Creates moderate collision risk but allows reaction time
- Object was somewhat predictable but still poses safety concern
- Distance allows for controlled response

**ANALYSIS REQUIREMENTS:**

For each 10-second interval, provide a JSON object with these fields:
- "timestamp": Time range (e.g., "0-10s")
- "summary": Brief scene description focusing on vehicle movement and object interactions
- "actions": Current actions of ego vehicle and other traffic participants
- "characters": People visible in the scene (if any)
- "key_objects": Important objects, vehicles, or infrastructure
- "key_actions": **CRITICAL FIELD** - Use "ghost probing" for high-confidence cases, "potential ghost probing" for moderate cases, or describe other significant actions
- "next_action": Predicted immediate next action of ego vehicle

**DETECTION GUIDELINES:**
- Focus on concealment and sudden appearance
- Consider collision risk and reaction time available
- Prioritize safety-critical situations
- Be specific about object types (vehicle, pedestrian, cyclist, etc.)
- Consider the ego vehicle's perspective and available sight lines

Analyze the provided frames and return a JSON array of interval analyses."""

def setup_gemini_with_manual_key():
    """手动输入API Key设置Gemini"""
    print("🔑 当前API Key配额已用完，请输入另一个可用的Gemini API Key")
    print("💡 您可以从Google AI Studio获取新的API Key: https://aistudio.google.com/app/apikey")
    
    api_key = getpass.getpass("请输入Gemini API Key (输入时不会显示): ").strip()
    
    if not api_key:
        print("❌ 未输入API Key")
        return None
    
    print(f"🔑 测试API Key: {api_key[:15]}...")
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # 测试API调用
        test_response = model.generate_content(
            "Test message - return just 'OK'",
            generation_config={"temperature": 0.1, "max_output_tokens": 10}
        )
        
        if test_response and test_response.text:
            print(f"✅ API Key 可用！响应: {test_response.text.strip()}")
            return model
        else:
            print("❌ API Key 响应为空")
            return None
            
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            print("⚠️ 该API Key配额已用完")
        else:
            print(f"❌ API Key 错误: {str(e)}")
        return None

def extract_frames(video_path, output_dir, interval=10, frames_per_interval=10):
    """提取视频帧"""
    if not os.path.exists(video_path):
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            return []
        
        duration = total_frames / fps
        intervals = int(duration // interval) + (1 if duration % interval > 0 else 0)
        
        frame_paths = []
        
        for i in range(intervals):
            start_time = i * interval
            end_time = min((i + 1) * interval, duration)
            interval_duration = end_time - start_time
            
            if interval_duration <= 0:
                continue
            
            frames_to_extract = min(frames_per_interval, max(1, int(interval_duration * fps)))
            
            for j in range(frames_to_extract):
                timestamp = start_time + (j * interval_duration / frames_to_extract)
                frame_number = int(timestamp * fps)
                
                if frame_number >= total_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    frame_filename = f"frame_{i:03d}_{j:03d}_{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
        
        cap.release()
        return frame_paths
        
    except Exception as e:
        print(f"帧提取错误: {e}")
        return []

def process_single_video(video_path, model, output_dir):
    """处理单个视频"""
    try:
        video_id = os.path.basename(video_path).replace('.avi', '')
        video_id_formatted = video_id.replace('images_', 'dada_')
        output_file = os.path.join(output_dir, f"actionSummary_{video_id_formatted}.json")
        
        if os.path.exists(output_file):
            print(f"⏭️ 跳过已处理的视频: {video_id}")
            return True
        
        print(f"🎬 处理视频: {video_id}")
        
        # 提取帧
        temp_frames_dir = f"frames_temp_{hash(video_path) % 100000}"
        frame_paths = extract_frames(video_path, temp_frames_dir)
        
        if not frame_paths:
            print(f"❌ 无法提取帧: {video_id}")
            return False
        
        # 准备图像
        images = []
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                with open(frame_path, 'rb') as f:
                    images.append({
                        'mime_type': 'image/jpeg',
                        'data': f.read()
                    })
        
        if not images:
            print(f"❌ 没有有效图像: {video_id}")
            return False
        
        # 调用Gemini
        prompt = get_balanced_prompt()
        
        response = model.generate_content(
            [prompt] + images,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 8192,
            }
        )
        
        if not response or not response.text:
            print(f"❌ API响应为空: {video_id}")
            return False
        
        # 解析JSON响应
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        try:
            result = json.loads(response_text)
            
            # 保存结果
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 成功处理: {video_id}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误 {video_id}: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 处理视频错误 {video_id}: {e}")
        return False
        
    finally:
        # 清理临时文件
        if 'temp_frames_dir' in locals() and os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir, ignore_errors=True)

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
    print("🚀 Gemini 手动API Key输入 - 完成剩余50个视频")
    print("=" * 60)
    
    # 检查剩余视频
    remaining_videos = get_remaining_videos()
    print(f"📋 剩余未处理视频: {len(remaining_videos)} 个")
    
    if len(remaining_videos) == 0:
        print("🎉 所有视频已处理完成！")
        return
    
    # 手动输入API Key设置Gemini
    model = setup_gemini_with_manual_key()
    if not model:
        print("❌ 无法初始化Gemini模型")
        return
    
    print(f"🎯 开始处理 {len(remaining_videos)} 个视频")
    print(f"📊 配额限制: 200 RPD (应该足够处理所有剩余视频)")
    
    # 处理视频
    output_dir = "result/gemini-balanced-full"
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failed_videos = []
    
    for i, video_path in enumerate(tqdm(remaining_videos, desc="处理视频"), 1):
        print(f"\n[{i}/{len(remaining_videos)}] 处理: {os.path.basename(video_path)}")
        
        if process_single_video(video_path, model, output_dir):
            success_count += 1
        else:
            failed_videos.append(video_path)
        
        # 每10个视频报告一次进度
        if i % 10 == 0:
            print(f"📊 进度更新: {success_count}/{i} 成功 ({success_count/i*100:.1f}%)")
        
        # 添加延迟以避免API限制
        time.sleep(2)
    
    # 最终统计
    total_processed = 49 + success_count  # 之前49个 + 新处理的
    
    print(f"\n🎯 处理完成!")
    print(f"  📊 本次成功: {success_count}/{len(remaining_videos)}")
    print(f"  📊 总体进度: {total_processed}/99 ({total_processed/99*100:.1f}%)")
    
    if failed_videos:
        print(f"  ❌ 失败视频: {len(failed_videos)} 个")
        for vid in failed_videos:
            print(f"    - {os.path.basename(vid)}")
    
    if total_processed >= 99:
        print("🎉 恭喜！已完成全部99个视频的Gemini处理！")
        print("📊 现在可以进行完整的99视频对比分析")
        
        # 自动启动完整对比分析
        print("\n🔄 自动启动完整99视频对比分析...")
        try:
            import subprocess
            subprocess.run(["python", "create_final_99_video_comparison.py"], check=False)
        except:
            print("💡 请手动运行: python create_final_99_video_comparison.py")
    else:
        print(f"⏳ 还需处理 {99 - total_processed} 个视频")

if __name__ == "__main__":
    main()