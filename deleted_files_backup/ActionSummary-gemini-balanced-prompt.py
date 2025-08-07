#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini 2.0 Flash + GPT-4.1平衡版Prompt 公平对比实验
基于ActionSummary-gemini.py，使用GPT-4.1的平衡版prompt进行公平对比
"""

import cv2
import os
import base64
import requests
from moviepy.editor import VideoFileClip
import logging
import json
import sys
import threading
from retrying import retry
logging.getLogger('moviepy').setLevel(logging.ERROR)
import time
from functools import wraps
from dotenv import load_dotenv
import time
import video_utilities as vu
from jinja2 import Environment, FileSystemLoader
import numpy as np
import tqdm
import traceback
import datetime
import multiprocessing
from functools import partial
import re

# Gemini相关库
import google.generativeai as genai

def process_video_wrapper(video_path, args):
    """视频处理的包装函数，用于多进程调用"""
    try:
        video_name = os.path.basename(video_path)
        result_filename = f'actionSummary_{video_name.split(".")[0]}.json'
        result_path = os.path.join(args.output_dir, result_filename)
        
        if os.path.exists(result_path) and not args.no_skip and not args.retry_failed:
            print(f"视频 {video_name} 已经处理过，跳过")
            return (video_path, "skipped", None, 0)
        
        processor = GeminiVideoProcessor(args)
        result = processor.process_video(video_path)
        
        if result is None:
            return (video_path, "failed", None, 0)
        
        # 保存结果
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return (video_path, "success", result, len(result))
        
    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {str(e)}")
        return (video_path, "failed", str(e), 0)

class GeminiVideoProcessor:
    def __init__(self, args):
        self.frame_interval = args.interval
        self.frames_per_interval = args.frames
        self.output_dir = args.output_dir
        self.max_retries = getattr(args, 'max_retries', 3)
        
        # 初始化Gemini
        load_dotenv()
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # 设置日志
        self.setup_logging()
    
    def setup_logging(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/gemini_balanced_processing_{timestamp}.log"
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_balanced_prompt(self, video_id, segment_id_str, frame_interval, frames_per_interval):
        """
        使用GPT-4.1平衡版的分层检测策略prompt
        """
        system_prompt = f"""You are an expert AI system analyzing sequential video frames from autonomous driving scenarios. Your primary task is to detect "ghost probing" events using a balanced layered detection strategy.

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

Your job is to analyze {frames_per_interval} frames spanning {frame_interval} seconds and provide detailed analysis.

**TASKS:**
1. **Ghost Probing Detection**: Apply the layered detection strategy
2. **Current Action Analysis**: Describe what's happening in the video
3. **Next Action Prediction**: Predict required vehicle response
4. **Object-Action Consistency**: Ensure key_objects match key_actions

Always return a single JSON object with these fields:
- video_id: "{video_id}"
- segment_id: "{segment_id_str}"
- Start_Timestamp and End_Timestamp: derived from frame names
- summary: detailed description of the scenario
- actions: current vehicle actions and reasoning
- key_objects: important objects affecting driving decisions
- key_actions: classification using layered strategy ("ghost probing", "potential ghost probing", or "none")
- next_action: JSON object with speed_control, direction_control, and lane_control

**IMPORTANT**: Use the layered detection strategy to maintain high recall (detect real dangers) while improving precision (reduce false positives). When in doubt between categories, prefer the more conservative classification.

All text must be in English. Return only valid JSON."""

        return system_prompt.replace("{video_id}", video_id).replace("{segment_id_str}", segment_id_str)
    
    def process_video_segment(self, video_path, start_time, end_time, video_id, segment_id):
        """处理视频片段"""
        try:
            # 提取帧
            image_paths = vu.extract_frames_at_intervals(
                video_path, start_time, end_time, self.frames_per_interval
            )
            
            if not image_paths:
                self.logger.error(f"未能提取帧: {video_path}")
                return None
            
            # 提取音频转录（简化版）
            trans = ""  # Gemini主要依靠视觉分析
            
            # 构建分析请求
            return self.analyze_with_gemini(image_paths, trans, video_id, f"segment_{segment_id:03d}")
            
        except Exception as e:
            self.logger.error(f"处理视频片段时出错: {str(e)}")
            return None
    
    def analyze_with_gemini(self, image_paths, trans, video_id, segment_id_str):
        """使用Gemini进行分析"""
        try:
            # 获取平衡版prompt
            system_prompt = self.get_balanced_prompt(
                video_id, segment_id_str, self.frame_interval, self.frames_per_interval
            )
            
            # 准备图像数据
            prompt_parts = []
            if trans:
                prompt_parts.append(f"Audio Transcription: {trans}")
            
            prompt_parts.append(f"Analyzing {len(image_paths)} frames from {self.frame_interval} seconds:")
            
            # 添加图像
            images = []
            for img_path in image_paths:
                try:
                    # 读取并编码图像
                    with open(img_path, 'rb') as f:
                        image_data = f.read()
                    
                    # Gemini图像格式
                    images.append({
                        'mime_type': 'image/jpeg',
                        'data': image_data
                    })
                    
                    prompt_parts.append(f"Frame: {os.path.basename(img_path)}")
                    
                except Exception as e:
                    self.logger.warning(f"无法加载图像 {img_path}: {str(e)}")
                    continue
            
            if not images:
                self.logger.error("没有成功加载任何图像")
                return None
            
            # 构建请求内容
            content = [system_prompt] + [{"text": part} for part in prompt_parts] + images
            
            # 安全设置
            safety_settings = [
                {"category": "harassment", "threshold": "block_only_high"},
                {"category": "hate_speech", "threshold": "block_only_high"},
                {"category": "sexually_explicit", "threshold": "block_only_high"},
                {"category": "dangerous_content", "threshold": "block_only_high"}
            ]
            
            # 生成配置
            generation_config = {
                "temperature": 0.2,  # 与GPT-4.1保持一致
                "top_p": 0.95,
                "max_output_tokens": 2048,
            }
            
            # API调用
            self.logger.info(f"调用Gemini API分析 {len(images)} 个图像")
            
            response = self.model.generate_content(
                content,
                safety_settings=safety_settings,
                generation_config=generation_config
            )
            
            if not response.text:
                self.logger.error("Gemini API返回空响应")
                return None
            
            # 解析JSON响应
            try:
                result = json.loads(response.text)
                self.logger.info("成功解析Gemini响应JSON")
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON解析失败: {str(e)}")
                self.logger.error(f"原始响应: {response.text[:500]}...")
                return None
                
        except Exception as e:
            self.logger.error(f"Gemini API调用失败: {str(e)}")
            return None
    
    def process_video(self, video_path):
        """处理整个视频"""
        try:
            self.logger.info(f"开始处理视频: {video_path}")
            
            # 获取视频信息
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
            
            video_name = os.path.basename(video_path)
            video_id = video_name.split('.')[0].replace('images_', 'dada_')
            
            # 计算分段
            total_segments = int(duration // self.frame_interval)
            if duration % self.frame_interval > 0:
                total_segments += 1
            
            results = []
            
            for segment_id in range(total_segments):
                start_time = segment_id * self.frame_interval
                end_time = min((segment_id + 1) * self.frame_interval, duration)
                
                self.logger.info(f"处理片段 {segment_id + 1}/{total_segments}: {start_time:.1f}s - {end_time:.1f}s")
                
                result = self.process_video_segment(
                    video_path, start_time, end_time, video_id, segment_id
                )
                
                if result:
                    results.append(result)
                    self.logger.info(f"片段 {segment_id + 1} 处理成功")
                else:
                    self.logger.warning(f"片段 {segment_id + 1} 处理失败")
                
                # 进度报告
                progress = (segment_id + 1) / total_segments * 100
                print(f"进度: {progress:.1f}% [{segment_id + 1}/{total_segments}]")
            
            self.logger.info(f"视频处理完成: {len(results)}/{total_segments} 片段成功")
            return results
            
        except Exception as e:
            self.logger.error(f"处理视频时出错: {str(e)}")
            return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Gemini 2.0 Flash + 平衡版Prompt 视频分析')
    parser.add_argument('--folder', default='DADA-2000-videos', help='视频文件夹路径')
    parser.add_argument('--single', help='处理单个视频文件')
    parser.add_argument('--output-dir', default='result/gemini-balanced-prompt', help='输出目录')
    parser.add_argument('--interval', type=int, default=10, help='时间间隔(秒)')
    parser.add_argument('--frames', type=int, default=10, help='每个间隔的帧数')
    parser.add_argument('--limit', type=int, help='限制处理的视频数量')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已处理的视频')
    parser.add_argument('--retry-failed', action='store_true', help='重新处理失败的视频')
    parser.add_argument('--start-at', type=int, default=0, help='从第几个视频开始')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取视频列表
    if args.single:
        video_files = [args.single]
    else:
        video_files = []
        for filename in sorted(os.listdir(args.folder)):
            if filename.endswith('.avi') and filename.startswith('images_'):
                video_files.append(os.path.join(args.folder, filename))
    
    # 过滤和限制
    if args.start_at > 0:
        video_files = video_files[args.start_at:]
    
    if args.limit:
        video_files = video_files[:args.limit]
    
    print(f"📊 准备处理 {len(video_files)} 个视频")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🔧 配置: {args.interval}秒间隔, {args.frames}帧/间隔")
    print("🚀 使用Gemini 2.0 Flash + GPT-4.1平衡版Prompt")
    
    # 处理视频
    successful = 0
    failed = 0
    skipped = 0
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n处理视频 {i}/{len(video_files)}: {os.path.basename(video_path)}")
        
        result = process_video_wrapper(video_path, args)
        video_path, status, data, segments = result
        
        if status == "success":
            successful += 1
            print(f"✅ 成功处理 {segments} 个片段")
        elif status == "skipped":
            skipped += 1
            print(f"⏭️ 跳过已处理")
        else:
            failed += 1
            print(f"❌ 处理失败: {data}")
    
    # 最终统计
    print(f"\n🎯 处理完成统计:")
    print(f"  ✅ 成功: {successful}")
    print(f"  ⏭️ 跳过: {skipped}")
    print(f"  ❌ 失败: {failed}")
    print(f"  📊 成功率: {successful/(successful+failed)*100:.1f}%")

if __name__ == "__main__":
    main()