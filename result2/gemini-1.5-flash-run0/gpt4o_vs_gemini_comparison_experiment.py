#!/usr/bin/env python3
"""
GPT-4o vs Gemini-1.5-flash 对比实验
重复Documentation中记录的实验，保证准确性
基于DADA-2000数据集进行ghost probing检测对比
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
# import video_utilities as vu  # 暂时注释掉，不是必需的
# from jinja2 import Environment, FileSystemLoader  # 暂时注释掉，不是必需的
import numpy as np
import tqdm
import traceback
import datetime
import multiprocessing
from functools import partial
import re
import pandas as pd
import google.generativeai as genai
import openai

# 加载环境变量
load_dotenv()

class ExperimentRunner:
    def __init__(self):
        self.setup_logging()
        self.setup_apis()
        self.load_ground_truth()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/experiment_log_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("实验开始 - GPT-4o vs Gemini-1.5-flash 对比")
        
    def setup_apis(self):
        """设置API配置"""
        # OpenAI GPT-4o配置
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.vision_api_type = os.environ.get("VISION_API_TYPE", "OpenAI")
        
        # Azure配置（如果使用）
        if self.vision_api_type == "Azure":
            self.azure_vision_key = os.environ.get("AZURE_VISION_KEY", "")
            self.vision_deployment = os.environ.get("VISION_DEPLOYMENT_NAME", "gpt-4o")
            self.vision_endpoint = os.environ.get("VISION_ENDPOINT", "")
        
        # Gemini配置
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.logger.warning("Gemini API Key未设置，Gemini测试将被跳过")
            self.gemini_model = None
        
        # 检查必要的API配置
        if not self.openai_api_key and self.vision_api_type == "OpenAI":
            self.logger.warning("OpenAI API Key未设置，GPT-4o测试可能失败")
        if not self.azure_vision_key and self.vision_api_type == "Azure":
            self.logger.warning("Azure Vision Key未设置，GPT-4o测试可能失败")
            
        self.logger.info("API配置完成")
        
    def load_ground_truth(self):
        """加载ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
        self.ground_truth = pd.read_csv(gt_path, sep='\t')
        self.logger.info(f"加载ground truth标签: {len(self.ground_truth)}个视频")
        
    def extract_frames_from_video(self, video_path, frame_interval=10, frames_per_interval=10):
        """从视频中提取帧"""
        frames_dir = "frames_temp"
        os.makedirs(frames_dir, exist_ok=True)
        
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            frames = []
            timestamps = []
            
            # 计算间隔时间点
            num_intervals = max(1, int(duration / frame_interval))
            
            for interval_idx in range(num_intervals):
                start_time = interval_idx * frame_interval
                end_time = min((interval_idx + 1) * frame_interval, duration)
                
                # 在每个间隔内均匀提取帧
                for frame_idx in range(frames_per_interval):
                    if frames_per_interval == 1:
                        frame_time = start_time + (end_time - start_time) / 2
                    else:
                        frame_time = start_time + (frame_idx / (frames_per_interval - 1)) * (end_time - start_time)
                    
                    if frame_time >= duration:
                        break
                        
                    frame = clip.get_frame(frame_time)
                    frame_filename = f"frame_{interval_idx}_{frame_idx}_{frame_time:.1f}s.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frames.append(frame_path)
                    timestamps.append(frame_time)
            
            clip.close()
            return frames, timestamps
            
        except Exception as e:
            self.logger.error(f"帧提取失败 {video_path}: {str(e)}")
            return [], []
            
    def encode_image_to_base64(self, image_path):
        """将图像编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def get_balanced_gpt4o_prompt(self, video_id, frame_interval, frames_per_interval):
        """获取平衡版GPT-4.1 prompt"""
        return f"""You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the video.

IMPORTANT: For ghost probing detection, consider TWO categories:

**1. HIGH-CONFIDENCE Ghost Probing (use "ghost probing" in key_actions)**:
- Object appears EXTREMELY close (within 1-2 vehicle lengths, <3 meters) 
- Appearance is SUDDEN and from blind spots (behind parked cars, buildings, corners)
- Occurs in HIGH-RISK environments: highways, rural roads, parking lots, uncontrolled intersections
- Requires IMMEDIATE emergency braking/swerving to avoid collision
- Movement is COMPLETELY UNPREDICTABLE and violates traffic expectations

**2. POTENTIAL Ghost Probing (use "potential ghost probing" in key_actions)**:
- Object appears suddenly but at moderate distance (3-5 meters)
- Sudden movement in environments where some unpredictability exists
- Requires emergency braking but collision risk is moderate
- Movement is unexpected but not completely impossible given the context

**3. NORMAL Traffic Situations (do NOT use "ghost probing")**:
- Pedestrians crossing at intersections, crosswalks, or traffic lights
- Vehicles making normal lane changes, turns, or merging with signals
- Cyclists following predictable paths in urban areas or bike lanes
- Any movement that is EXPECTED given the traffic environment and context

**Environment Context Guidelines**:
- INTERSECTION/CROSSWALK: Expect pedestrians and cyclists - use "emergency braking due to pedestrian crossing"
- HIGHWAY/RURAL: Higher chance of genuine ghost probing - be more sensitive
- PARKING LOT: Expect sudden vehicle movements - use "potential ghost probing" if very sudden
- URBAN STREET: Mixed - consider visibility and predictability

Use "ghost probing" for clear cases, "potential ghost probing" for borderline cases, and descriptive terms for normal traffic situations.

Your response should be a valid JSON object with the following EXACT structure:
{{
    "video_id": "{video_id}",
    "segment_id": "full_video",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "{frame_interval}.0s",
    "sentiment": "Positive/Negative/Neutral",
    "scene_theme": "Dramatic/Routine/Dangerous/Safe",
    "characters": "brief description of people in the scene",
    "summary": "comprehensive summary of the scene and what happens",
    "actions": "actions taken by the vehicle and driver responses",
    "key_objects": "numbered list: 1) Position: object description, distance, behavior impact 2) Position: object description, distance, behavior impact",
    "key_actions": "brief description of most important actions (use 'ghost probing', 'potential ghost probing', or descriptive terms as appropriate)",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}"""

    def analyze_with_gpt4o(self, video_path, video_id):
        """使用GPT-4o分析视频"""
        # 检查API配置
        if self.vision_api_type == "OpenAI" and not self.openai_api_key:
            self.logger.error("OpenAI API Key未配置，跳过GPT-4o分析")
            return None
        if self.vision_api_type == "Azure" and not self.azure_vision_key:
            self.logger.error("Azure Vision Key未配置，跳过GPT-4o分析")
            return None
            
        try:
            frames, timestamps = self.extract_frames_from_video(video_path)
            if not frames:
                return None
                
            # 编码图像
            base64_images = []
            for frame_path in frames:
                base64_image = self.encode_image_to_base64(frame_path)
                base64_images.append(base64_image)
                
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.get_balanced_gpt4o_prompt(video_id, 10, 10)}
                    ] + [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                        for img in base64_images
                    ]
                }
            ]
            
            # API调用
            if self.vision_api_type == "Azure":
                client = openai.AzureOpenAI(
                    api_key=self.azure_vision_key,
                    api_version="2024-02-01",
                    azure_endpoint=f"https://{self.vision_endpoint}.openai.azure.com/"
                )
                response = client.chat.completions.create(
                    model=self.vision_deployment,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0
                )
            else:
                client = openai.OpenAI(api_key=self.openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=1000,
                    temperature=0
                )
                
            result = response.choices[0].message.content
            
            # 清理临时文件
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"GPT-4o分析失败 {video_id}: {str(e)}")
            return None
            
    def analyze_with_gemini(self, video_path, video_id):
        """使用Gemini-1.5-flash分析视频"""
        # 检查API配置
        if not self.gemini_model:
            self.logger.error("Gemini API未配置，跳过Gemini分析")
            return None
            
        try:
            frames, timestamps = self.extract_frames_from_video(video_path)
            if not frames:
                return None
                
            # 准备图像
            images = []
            for frame_path in frames:
                with open(frame_path, 'rb') as f:
                    image_data = f.read()
                images.append({
                    'mime_type': 'image/jpeg',
                    'data': image_data
                })
                
            # 构建prompt (使用相同的prompt但适配Gemini格式)
            prompt = self.get_balanced_gpt4o_prompt(video_id, 10, 10)
            
            # API调用
            response = self.gemini_model.generate_content([prompt] + images)
            result = response.text
            
            # 清理临时文件
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Gemini分析失败 {video_id}: {str(e)}")
            return None
    
    def extract_key_actions(self, result_text):
        """从结果中提取key_actions"""
        try:
            # 尝试解析JSON
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            result_json = json.loads(result_text)
            return result_json.get('key_actions', '').lower()
        except:
            # 如果JSON解析失败，使用正则表达式
            key_actions_match = re.search(r'"key_actions":\s*"([^"]*)"', result_text)
            if key_actions_match:
                return key_actions_match.group(1).lower()
            return result_text.lower()
    
    def evaluate_result(self, video_id, key_actions, ground_truth_label):
        """评估结果"""
        has_ghost_probing = "ghost probing" in key_actions or "potential ghost probing" in key_actions
        ground_truth_has_ghost = ground_truth_label != "none"
        
        if has_ghost_probing and ground_truth_has_ghost:
            return "TP"  # True Positive
        elif has_ghost_probing and not ground_truth_has_ghost:
            return "FP"  # False Positive
        elif not has_ghost_probing and ground_truth_has_ghost:
            return "FN"  # False Negative
        else:
            return "TN"  # True Negative
    
    def run_experiment(self, video_limit=20):
        """运行实验"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "experiment_info": {
                "timestamp": timestamp,
                "video_limit": video_limit,
                "models": ["gpt-4o", "gemini-1.5-flash"]
            },
            "results": []
        }
        
        # 选择前N个视频进行测试
        test_videos = self.ground_truth.head(video_limit)
        
        self.logger.info(f"开始处理 {len(test_videos)} 个视频")
        
        for idx, row in tqdm.tqdm(test_videos.iterrows(), total=len(test_videos)):
            video_id = row['video_id']
            ground_truth_label = row['ground_truth_label']
            
            video_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/{video_id}"
            
            if not os.path.exists(video_path):
                self.logger.warning(f"视频文件不存在: {video_path}")
                continue
                
            self.logger.info(f"处理视频 {idx+1}/{len(test_videos)}: {video_id}")
            
            # GPT-4o分析
            gpt4o_result = self.analyze_with_gpt4o(video_path, video_id)
            gpt4o_key_actions = ""
            gpt4o_evaluation = "ERROR"
            
            if gpt4o_result:
                gpt4o_key_actions = self.extract_key_actions(gpt4o_result)
                gpt4o_evaluation = self.evaluate_result(video_id, gpt4o_key_actions, ground_truth_label)
            
            # Gemini分析
            gemini_result = self.analyze_with_gemini(video_path, video_id)
            gemini_key_actions = ""
            gemini_evaluation = "ERROR"
            
            if gemini_result:
                gemini_key_actions = self.extract_key_actions(gemini_result)
                gemini_evaluation = self.evaluate_result(video_id, gemini_key_actions, ground_truth_label)
            
            # 记录结果
            video_result = {
                "video_id": video_id,
                "ground_truth": ground_truth_label,
                "gpt4o": {
                    "raw_result": gpt4o_result,
                    "key_actions": gpt4o_key_actions,
                    "evaluation": gpt4o_evaluation
                },
                "gemini": {
                    "raw_result": gemini_result,
                    "key_actions": gemini_key_actions,
                    "evaluation": gemini_evaluation
                }
            }
            
            results["results"].append(video_result)
            
            # 实时保存结果
            result_file = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gpt4o_vs_gemini_results_{timestamp}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        self.generate_summary_report(results, timestamp)
        
    def generate_summary_report(self, results, timestamp):
        """生成总结报告"""
        # 计算性能指标
        gpt4o_stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "ERROR": 0}
        gemini_stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "ERROR": 0}
        
        for result in results["results"]:
            gpt4o_stats[result["gpt4o"]["evaluation"]] += 1
            gemini_stats[result["gemini"]["evaluation"]] += 1
            
        def calculate_metrics(stats):
            tp, fp, tn, fn = stats["TP"], stats["FP"], stats["TN"], stats["FN"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "stats": stats
            }
        
        gpt4o_metrics = calculate_metrics(gpt4o_stats)
        gemini_metrics = calculate_metrics(gemini_stats)
        
        # 生成报告
        report = {
            "experiment_summary": {
                "timestamp": timestamp,
                "total_videos": len(results["results"]),
                "models_compared": ["GPT-4o", "Gemini-1.5-flash"]
            },
            "performance_comparison": {
                "gpt4o": gpt4o_metrics,
                "gemini": gemini_metrics
            },
            "detailed_results": results["results"]
        }
        
        # 保存报告
        report_file = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/experiment_summary_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        # 打印总结
        print("\n" + "="*50)
        print("实验总结报告")
        print("="*50)
        print(f"测试视频数量: {len(results['results'])}")
        print(f"实验时间: {timestamp}")
        print("\nGPT-4o 性能:")
        print(f"  精确度: {gpt4o_metrics['precision']:.3f}")
        print(f"  召回率: {gpt4o_metrics['recall']:.3f}")
        print(f"  F1分数: {gpt4o_metrics['f1_score']:.3f}")
        print(f"  准确率: {gpt4o_metrics['accuracy']:.3f}")
        print(f"  统计: {gpt4o_metrics['stats']}")
        
        print("\nGemini-1.5-flash 性能:")
        print(f"  精确度: {gemini_metrics['precision']:.3f}")
        print(f"  召回率: {gemini_metrics['recall']:.3f}")
        print(f"  F1分数: {gemini_metrics['f1_score']:.3f}")
        print(f"  准确率: {gemini_metrics['accuracy']:.3f}")
        print(f"  统计: {gemini_metrics['stats']}")
        
        print(f"\n详细结果已保存到: {report_file}")
        
        self.logger.info("实验完成")

if __name__ == "__main__":
    runner = ExperimentRunner()
    
    # 运行实验，首先用少量视频测试
    video_limit = 20  # 可以调整测试视频数量
    
    print(f"开始 GPT-4o vs Gemini-1.5-flash 对比实验")
    print(f"测试视频数量: {video_limit}")
    print("="*50)
    
    runner.run_experiment(video_limit=video_limit)