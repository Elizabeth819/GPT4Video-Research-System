#!/usr/bin/env python3
"""
Run 8 Supplement: 处理images_5_055.avi (DADA-2000数据集)
使用与Run 8完全相同的配置和prompt处理第100个视频
"""

import cv2
import os
import json
import logging
import time
import datetime
from moviepy.editor import VideoFileClip
import pandas as pd
from dotenv import load_dotenv
import base64
import requests
import traceback

# 加载环境变量
load_dotenv()

class Run8ProcessImages5055:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logging()
        self.setup_openai_api()
        self.initialize_results()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        log_filename = os.path.join(self.output_dir, f"run8_images_5_055_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Run 8: 处理images_5_055.avi开始")
        
    def setup_openai_api(self):
        """设置OpenAI API"""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY未设置")
        
        # Azure OpenAI配置
        self.vision_endpoint = os.environ.get("VISION_ENDPOINT", "")
        self.vision_deployment = os.environ.get("VISION_DEPLOYMENT_NAME", "gpt-4o-global")
        
        if not self.vision_endpoint:
            raise ValueError("VISION_ENDPOINT未设置")
            
        self.logger.info(f"Azure OpenAI API配置成功")
        self.logger.info(f"Endpoint: {self.vision_endpoint}")
        self.logger.info(f"Deployment: {self.vision_deployment}")
        self.logger.info(f"Temperature: 0 (与Run 8完全一致)")
        
    def initialize_results(self):
        """初始化结果结构"""
        self.results = {
            "experiment_info": {
                "run_id": "Run 8 - images_5_055",
                "timestamp": self.timestamp,
                "video_count": 1,
                "model": "GPT-4o (Azure)",
                "prompt_version": "Paper_Batch Complex (4-Task) + Few-shot Examples",
                "temperature": 0,
                "max_tokens": 3000,
                "purpose": "处理DADA-2000中的images_5_055.avi补充Run 8到100视频",
                "output_directory": self.output_dir,
                "base_configuration": "与Run 8完全一致的参数和prompt配置"
            },
            "detailed_results": []
        }
        
    def extract_frames_from_video(self, video_path, frame_interval=10, frames_per_interval=10):
        """从视频中提取帧"""
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            frames = []
            frames_dir = os.path.join(self.output_dir, "frames_temp")
            os.makedirs(frames_dir, exist_ok=True)
            
            # 计算间隔数
            num_intervals = max(1, int(duration / frame_interval))
            
            for interval_idx in range(num_intervals):
                start_time = interval_idx * frame_interval
                end_time = min((interval_idx + 1) * frame_interval, duration)
                
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
            
            clip.close()
            return frames
        except Exception as e:
            self.logger.error(f"帧提取失败 {video_path}: {str(e)}")
            return []
    
    def get_run8_paper_batch_fewshot_prompt(self, video_id, frame_interval=10, frames_per_interval=10):
        """获取与Run 8完全一致的Paper_Batch + Few-shot prompt"""
        return f'''You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the video.

**Few-shot Examples for Ghost Probing Detection:**

**Example 1 - TRUE Ghost Probing (Rural Road):**
- Scene: Rural road, child suddenly runs from behind parked car
- Distance: <2 meters when first visible
- Timing: Appears at 3s, requires immediate braking
- Key_actions: "ghost probing"

**Example 2 - TRUE Ghost Probing (Urban Night):**
- Scene: Urban street at night, pedestrian falls directly in vehicle path
- Distance: Within 1-2 meters, sudden appearance
- Timing: Appears at 2s, requires emergency response
- Key_actions: "ghost probing"

**Example 3 - FALSE Positive (Normal Crossing):**
- Scene: Urban intersection with traffic light
- Behavior: Pedestrian crosses at designated crosswalk
- Visibility: Pedestrian visible for 5+ seconds before crossing
- Key_actions: "emergency braking due to pedestrian crossing"

**Example 4 - TRUE Ghost Probing (Highway):**
- Scene: Highway with vehicles and cyclist
- Distance: Cyclist appears suddenly from blind spot <3 meters
- Timing: Requires immediate lane change or braking
- Key_actions: "ghost probing"

**Classification Guidelines:**
Use "ghost probing" ONLY when ALL criteria are met:
1. Object appears suddenly (<3 meters away when first clearly visible)
2. From unexpected location (blind spot, behind obstruction, from side)
3. Requires immediate emergency response (rapid deceleration/swerving)
4. Movement is completely unpredictable given the context
5. High collision risk without immediate action

For borderline cases or normal traffic situations, use descriptive terms like:
- "emergency braking due to pedestrian crossing"
- "cautious driving due to cyclist presence"
- "emergency braking due to child running into street"
- "rapid deceleration for safety"

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
    "key_actions": "brief description of most important actions (use 'ghost probing' only for true cases matching all criteria)",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}

Audio Transcription: [No audio available for this analysis]

Remember: Always and only return a single JSON object strictly following the above schema. Use "ghost probing" only for genuine cases that meet ALL five criteria above.'''
    
    def send_azure_openai_request(self, prompt, images):
        """发送Azure OpenAI请求 - 使用Temperature=0与Run 8保持一致"""
        encoded_images = []
        for image_path in images:
            try:
                with open(image_path, 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    encoded_images.append(encoded_string)
            except Exception as e:
                self.logger.error(f"图像编码失败 {image_path}: {str(e)}")
                continue
        
        if not encoded_images:
            return None
            
        content = [{"type": "text", "text": prompt}]
        
        for encoded_image in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })
        
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 3000,  # 与Run 8保持一致
            "temperature": 0     # 与Run 8保持一致
        }
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.openai_api_key
        }
        
        try:
            response = requests.post(
                f"{self.vision_endpoint}/openai/deployments/{self.vision_deployment}/chat/completions?api-version=2024-02-01",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            self.logger.error(f"API调用失败: {str(e)}")
            return None
    
    def analyze_with_gpt4o(self, video_path, video_id):
        """使用GPT-4o分析视频（与Run 8相同配置）"""
        try:
            # 提取帧
            frames = self.extract_frames_from_video(video_path)
            if not frames:
                return None
            
            # 生成prompt
            prompt = self.get_run8_paper_batch_fewshot_prompt(video_id)
            
            # 发送API请求
            result = self.send_azure_openai_request(prompt, frames)
            
            # 清理临时帧文件
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            
            return result
        except Exception as e:
            self.logger.error(f"视频分析失败 {video_id}: {str(e)}")
            return None
    
    def extract_key_actions(self, result_text):
        """提取key_actions"""
        try:
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            result_json = json.loads(result_text)
            return result_json.get('key_actions', '').lower()
        except:
            # 尝试正则表达式提取
            import re
            key_actions_match = re.search(r'"key_actions":\s*"([^"]*)"', result_text)
            if key_actions_match:
                return key_actions_match.group(1).lower()
            return result_text.lower()
    
    def evaluate_result(self, video_id, key_actions, ground_truth_label):
        """评估结果 - 使用与Run 8相同的评估逻辑"""
        has_ghost_probing = "ghost probing" in key_actions
        ground_truth_has_ghost = ground_truth_label != "none"
        
        if has_ghost_probing and ground_truth_has_ghost:
            return "TP"
        elif has_ghost_probing and not ground_truth_has_ghost:
            return "FP"
        elif not has_ghost_probing and ground_truth_has_ghost:
            return "FN"
        else:
            return "TN"
    
    def process_images_5_055(self, ground_truth_label="unknown"):
        """处理images_5_055.avi"""
        video_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos/images_5_055.avi"
        video_id = "images_5_055.avi"
        
        try:
            self.logger.info(f"开始处理视频: {video_id}")
            
            if not os.path.exists(video_path):
                self.logger.error(f"视频文件不存在: {video_path}")
                return None
            
            # 分析视频
            result = self.analyze_with_gpt4o(video_path, video_id)
            
            if result:
                key_actions = self.extract_key_actions(result)
                evaluation = self.evaluate_result(video_id, key_actions, ground_truth_label)
            else:
                key_actions = ""
                evaluation = "ERROR"
            
            # 记录结果
            result_entry = {
                "video_id": video_id,
                "ground_truth": ground_truth_label,
                "key_actions": key_actions,
                "evaluation": evaluation,
                "raw_result": result
            }
            
            self.results["detailed_results"].append(result_entry)
            
            self.logger.info(f"视频 {video_id}: GT={ground_truth_label}, 检测={key_actions}, 评估={evaluation}")
            
            return result_entry
            
        except Exception as e:
            self.logger.error(f"处理视频失败 {video_id}: {str(e)}")
            return None
    
    def save_results(self):
        """保存结果"""
        result_file = os.path.join(self.output_dir, f"run8_images_5_055_results_{self.timestamp}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"结果已保存: {result_file}")
        return result_file

def main():
    # 配置
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8_gpt4o_ghost_probing_fewshot_100videos_results/images_5_055_supplement"
    
    # 创建处理器
    processor = Run8ProcessImages5055(output_dir)
    
    print(f"🎯 Run 8: 处理images_5_055.avi")
    print(f"📁 来源: DADA-2000-videos数据集")
    print(f"📂 输出目录: {output_dir}")
    print("=" * 50)
    
    # 处理视频
    start_time = time.time()
    # 由于这是DADA数据集的视频，我们假设它可能包含ghost probing，设为"unknown"等待分析
    result = processor.process_images_5_055("unknown")
    end_time = time.time()
    
    if result:
        # 保存结果
        result_file = processor.save_results()
        
        print(f"✅ 处理成功!")
        print(f"⏱️ 耗时: {end_time - start_time:.1f}秒")
        print(f"📁 结果文件: {result_file}")
        print(f"🔍 检测结果: {result['key_actions']}")
        print(f"📊 评估: {result['evaluation']}")
        print("")
        print("🎉 Run 8现在已处理100个视频 (99个DADA-100 + 1个DADA-2000)!")
        print("📊 这个结果将纳入Run 8的完整评估中")
    else:
        print("❌ 处理失败")

if __name__ == "__main__":
    main()