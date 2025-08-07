#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
重新处理失败的视频
"""

import os
import json
import cv2
import google.generativeai as genai
from dotenv import load_dotenv
import time
import shutil

def get_balanced_prompt():
    """获取平衡版prompt"""
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

def setup_gemini():
    """设置Gemini模型"""
    load_dotenv()
    
    # 尝试两个API Key
    api_keys = [
        ("GEMINI_API_KEY_2", os.getenv('GEMINI_API_KEY_2')),
        ("GEMINI_API_KEY", os.getenv('GEMINI_API_KEY'))
    ]
    
    for key_name, api_key in api_keys:
        if api_key:
            print(f"🔑 尝试 {key_name}: {api_key[:15]}...")
            genai.configure(api_key=api_key)
            
            try:
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                
                response = model.generate_content(
                    "Test",
                    generation_config={"temperature": 0.1, "max_output_tokens": 5}
                )
                
                if response and response.text:
                    print(f"✅ {key_name} 可用！")
                    return model
                    
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"⚠️ {key_name} 配额已用完")
                else:
                    print(f"❌ {key_name} 错误: {str(e)}")
                continue
    
    print("❌ 所有API Key都不可用")
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
        
        print(f"🎬 重新处理视频: {video_id}")
        
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

def main():
    print("🔄 重新处理失败的视频")
    print("=" * 40)
    
    # 失败的视频列表
    failed_videos = [
        "DADA-2000-videos/images_5_043.avi",
        "DADA-2000-videos/images_5_054.avi"
    ]
    
    # 检查哪些还需要处理
    output_dir = "result/gemini-balanced-full"
    remaining_failed = []
    
    for video_path in failed_videos:
        if os.path.exists(video_path):
            video_id = os.path.basename(video_path).replace('.avi', '').replace('images_', 'dada_')
            output_file = os.path.join(output_dir, f"actionSummary_{video_id}.json")
            if not os.path.exists(output_file):
                remaining_failed.append(video_path)
    
    print(f"📋 需要重新处理的视频: {len(remaining_failed)} 个")
    
    if len(remaining_failed) == 0:
        print("🎉 所有失败的视频都已完成！")
        return
    
    # 设置Gemini
    model = setup_gemini()
    if not model:
        print("❌ 无法获取可用的API Key")
        print("💡 建议等待几分钟后重试，或等待明天配额重置")
        return
    
    # 处理失败的视频
    success_count = 0
    
    for i, video_path in enumerate(remaining_failed, 1):
        print(f"\n[{i}/{len(remaining_failed)}] 处理: {os.path.basename(video_path)}")
        
        if process_single_video(video_path, model, output_dir):
            success_count += 1
            print(f"🎯 成功修复: {os.path.basename(video_path)}")
        else:
            print(f"❌ 仍然失败: {os.path.basename(video_path)}")
        
        # 添加更长的延迟避免配额问题
        if i < len(remaining_failed):
            print("⏳ 等待60秒避免配额限制...")
            time.sleep(60)
    
    # 检查最终结果
    total_processed = 97 + success_count
    print(f"\n🎯 重试结果:")
    print(f"  📊 本次成功: {success_count}/{len(remaining_failed)}")
    print(f"  📊 总体进度: {total_processed}/99 ({total_processed/99*100:.1f}%)")
    
    if total_processed >= 99:
        print("🎉 恭喜！全部99个视频处理完成！")
        print("📊 现在可以进行完整的99视频对比分析")

if __name__ == "__main__":
    main()