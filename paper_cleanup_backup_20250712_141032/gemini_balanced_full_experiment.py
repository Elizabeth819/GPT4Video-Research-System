#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini 2.0 Flash + 平衡版Prompt 完整实验 (99个视频)
基于快速测试的成功结果，扩展到全部Ground Truth视频
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
import traceback

def setup_gemini():
    """设置Gemini API"""
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ 请设置GEMINI_API_KEY环境变量")
        return None
    
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Gemini 2.0 Flash模型初始化成功")
        return model
    except Exception as e:
        print(f"❌ Gemini模型初始化失败: {str(e)}")
        return None

def get_balanced_prompt():
    """GPT-4.1平衡版prompt"""
    return """You are an expert AI system analyzing sequential video frames from autonomous driving scenarios. Your primary task is to detect "ghost probing" events using a balanced layered detection strategy.

**DEFINITION: Ghost Probing**
A dangerous traffic scenario where pedestrians, cyclists, or objects suddenly appear from concealed positions (behind parked cars, walls, blind spots) creating immediate collision risk requiring emergency braking or avoidance.

**LAYERED DETECTION STRATEGY:**

**1. HIGH-CONFIDENCE Ghost Probing (use "ghost probing" in key_actions)**:
- Object appears EXTREMELY close (within 1-2 vehicle lengths, <3 meters) 
- Appearance is SUDDEN and from blind spots (behind parked cars, walls, obstacles)
- Occurs in HIGH-RISK environments: highways, rural roads, parking lots
- Creates IMMEDIATE danger requiring emergency response
- Object was previously completely hidden and suddenly emerges

**2. POTENTIAL Ghost Probing (use "potential ghost probing" in key_actions)**:
- Object appears suddenly but at moderate distance (3-5 meters)
- Sudden movement in environments where some unpredictability exists
- Appears from partially concealed positions
- Creates heightened caution but not immediate emergency

**3. Normal Traffic (use "none" in key_actions)**:
- Predictable pedestrian crossings at crosswalks
- Cyclists in designated bike lanes
- Normal traffic flow and lane changes
- Expected movements in urban environments

**ANALYSIS FRAMEWORK:**
1. **Concealment Assessment**: Was the object previously hidden behind obstacles?
2. **Distance Evaluation**: How close is the object when first detected?
3. **Environment Context**: Is this a high-risk scenario location?
4. **Predictability**: Was this movement expected or sudden?
5. **Emergency Level**: Does this require immediate evasive action?

Always return a single JSON object with these fields:
- video_id: (extract from video filename)
- segment_id: (e.g., "segment_000")
- Start_Timestamp and End_Timestamp: derived from frame timing
- summary: detailed description of the scenario
- actions: current vehicle actions and reasoning
- key_objects: important objects affecting driving decisions
- key_actions: classification using layered strategy ("ghost probing", "potential ghost probing", or "none")
- next_action: JSON object with speed_control, direction_control, and lane_control

**IMPORTANT**: Use the layered detection strategy to maintain high recall (detect real dangers) while improving precision (reduce false positives). When in doubt between categories, prefer the more conservative classification.

All text must be in English. Return only valid JSON."""

def extract_frames(video_path, start_time, end_time, num_frames=10):
    """从视频中提取帧"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        duration = end_time - start_time
        
        for i in range(num_frames):
            timestamp = start_time + (i * duration / num_frames)
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # 确保frames目录存在
                os.makedirs("frames", exist_ok=True)
                frame_path = f"frames/temp_frame_{i:03d}.jpg"
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
        
        cap.release()
        return frames
        
    except Exception as e:
        print(f"❌ 帧提取失败: {str(e)}")
        return []

def analyze_with_gemini(model, frames, video_id, segment_id):
    """使用Gemini分析帧"""
    try:
        prompt = get_balanced_prompt()
        
        # 准备内容
        content = [prompt]
        content.append(f"\nAnalyzing video {video_id}, {segment_id}")
        content.append(f"Processing {len(frames)} frames:")
        
        # 添加图片
        for i, frame_path in enumerate(frames):
            if os.path.exists(frame_path):
                with open(frame_path, 'rb') as f:
                    image_data = f.read()
                
                content.append({
                    'mime_type': 'image/jpeg',
                    'data': image_data
                })
                content.append(f"Frame {i+1}: {os.path.basename(frame_path)}")
        
        # API调用
        response = model.generate_content(
            content,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            }
        )
        
        if response.text:
            return response.text
        else:
            return None
            
    except Exception as e:
        print(f"❌ Gemini分析失败: {str(e)}")
        return None

def parse_gemini_response(result_text):
    """解析Gemini响应为JSON"""
    try:
        # 清理可能的代码块标记
        clean_text = result_text.strip()
        if clean_text.startswith('```'):
            lines = clean_text.split('\n')
            # 找到第一个{和最后一个}
            json_start = -1
            json_end = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('{') and json_start == -1:
                    json_start = i
                if line.strip().endswith('}'):
                    json_end = i
            
            if json_start != -1 and json_end != -1:
                clean_text = '\n'.join(lines[json_start:json_end+1])
        
        return json.loads(clean_text)
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {str(e)}")
        return None

def process_video(model, video_path, output_dir):
    """处理单个视频"""
    video_name = os.path.basename(video_path)
    video_id = video_name.replace('.avi', '').replace('images_', 'dada_')
    
    # 检查是否已处理
    output_file = os.path.join(output_dir, f"actionSummary_{video_id}.json")
    if os.path.exists(output_file):
        return "skipped"
    
    try:
        # 获取视频信息
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
        
        # 计算片段数量
        interval = 10  # 10秒间隔
        num_segments = max(1, int(duration // interval))
        if duration % interval > 0:
            num_segments += 1
        
        results = []
        
        for seg_id in range(num_segments):
            start_time = seg_id * interval
            end_time = min((seg_id + 1) * interval, duration)
            
            # 提取帧
            frames = extract_frames(video_path, start_time, end_time, 10)
            
            if not frames:
                continue
            
            # Gemini分析
            result_text = analyze_with_gemini(model, frames, video_id, f"segment_{seg_id:03d}")
            
            if result_text:
                result_json = parse_gemini_response(result_text)
                if result_json:
                    results.append(result_json)
            
            # 清理临时帧
            for frame in frames:
                if os.path.exists(frame):
                    os.remove(frame)
            
            # 短暂延迟，避免API限制
            time.sleep(1)
        
        # 保存结果
        if results:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            return "success"
        else:
            return "failed"
            
    except Exception as e:
        print(f"❌ 处理视频 {video_name} 失败: {str(e)}")
        return "failed"

def get_ground_truth_videos():
    """获取Ground Truth视频列表"""
    try:
        df = pd.read_csv('result/groundtruth_labels.csv', sep='\t')
        video_ids = df['video_id'].str.replace('.avi', '').tolist()
        
        # 过滤掉空值
        video_ids = [vid for vid in video_ids if vid and isinstance(vid, str)]
        
        # 构建完整路径
        video_paths = []
        for vid in video_ids:
            video_path = f"DADA-2000-videos/{vid}.avi"
            if os.path.exists(video_path):
                video_paths.append(video_path)
        
        print(f"📊 找到 {len(video_paths)} 个Ground Truth视频")
        return video_paths
        
    except Exception as e:
        print(f"❌ 读取Ground Truth失败: {str(e)}")
        return []

def main():
    print("🚀 Gemini 2.0 Flash + 平衡版Prompt 完整实验")
    print("=" * 60)
    print("📋 目标: 处理99个Ground Truth视频，与GPT-4.1进行公平对比")
    
    # 初始化Gemini
    model = setup_gemini()
    if not model:
        return
    
    # 创建输出目录
    output_dir = "result/gemini-balanced-full"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频列表
    video_paths = get_ground_truth_videos()
    if not video_paths:
        print("❌ 未找到可处理的视频")
        return
    
    print(f"🎬 准备处理 {len(video_paths)} 个视频")
    print(f"📁 输出目录: {output_dir}")
    
    # 处理统计
    successful = 0
    skipped = 0
    failed = 0
    
    # 处理视频
    with tqdm(video_paths, desc="处理视频") as pbar:
        for video_path in pbar:
            video_name = os.path.basename(video_path)
            pbar.set_description(f"处理 {video_name}")
            
            try:
                result = process_video(model, video_path, output_dir)
                
                if result == "success":
                    successful += 1
                    pbar.write(f"✅ {video_name}")
                elif result == "skipped":
                    skipped += 1
                    pbar.write(f"⏭️ {video_name} (已处理)")
                else:
                    failed += 1
                    pbar.write(f"❌ {video_name}")
                
                # 更新进度条
                pbar.set_postfix({
                    '成功': successful,
                    '跳过': skipped, 
                    '失败': failed
                })
                
            except Exception as e:
                failed += 1
                pbar.write(f"❌ {video_name}: {str(e)}")
    
    # 最终统计
    total_processed = successful + failed
    success_rate = (successful / total_processed * 100) if total_processed > 0 else 0
    
    print(f"\n🎯 处理完成统计:")
    print(f"  ✅ 成功: {successful}")
    print(f"  ⏭️ 跳过: {skipped}")
    print(f"  ❌ 失败: {failed}")
    print(f"  📊 成功率: {success_rate:.1f}%")
    
    if successful > 0:
        print(f"\n📁 结果保存在: {output_dir}")
        print(f"📋 下一步: 运行公平对比分析")
    
    # 保存处理日志
    log_file = f"{output_dir}/processing_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
    log_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_videos': len(video_paths),
        'successful': successful,
        'skipped': skipped,
        'failed': failed,
        'success_rate': success_rate,
        'output_directory': output_dir
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"📝 处理日志已保存: {log_file}")

if __name__ == "__main__":
    main()