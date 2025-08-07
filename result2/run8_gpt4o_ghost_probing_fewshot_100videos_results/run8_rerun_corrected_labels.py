#!/usr/bin/env python3
"""
Run 8 重新运行: GPT-4o Ghost Probing Detection with Few-shot Learning (100 Videos)
完全相同的配置参数和prompt，基于校正后的labels.csv重新运行
目的：验证性能一致性和可能的提升
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
import tqdm
import re
import base64
import requests
import traceback
import sys

# 加载环境变量
load_dotenv()

class GPT4oRun8RerunCorrectedLabels:
    def __init__(self, output_dir, chunk_size=10):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logging()
        self.setup_openai_api()
        self.load_ground_truth()
        self.initialize_results()
        self.load_existing_results()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        log_filename = os.path.join(self.output_dir, f"run8_rerun_corrected_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_openai_api(self):
        """设置Azure OpenAI API"""
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.vision_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
        self.vision_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([self.openai_api_key, self.vision_endpoint, self.vision_deployment]):
            raise ValueError("Azure OpenAI环境变量未设置完整")
            
        self.logger.info(f"Azure OpenAI API配置成功")
        self.logger.info(f"Endpoint: {self.vision_endpoint}")
        self.logger.info(f"Deployment: {self.vision_deployment}")
        self.logger.info(f"Temperature: 0, Enhanced with Few-shot Examples")
        
    def load_ground_truth(self):
        """加载校正后的ground truth标签"""
        # 使用校正后的labels.csv文件
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/labels.csv"
        self.ground_truth = pd.read_csv(gt_path)
        self.logger.info(f"加载校正后的ground truth标签: {len(self.ground_truth)}个视频")
        
    def initialize_results(self):
        """初始化结果结构"""
        self.results = {
            "experiment_info": {
                "run_id": "Run 8 Rerun",
                "timestamp": self.timestamp,
                "video_count": 100,
                "model": "GPT-4o (Azure)",
                "prompt_version": "Paper_Batch Complex (4-Task) + Few-shot Examples",
                "temperature": 0,
                "max_tokens": 3000,
                "purpose": "重新运行验证：完整100视频测试最优Few-shot配置",
                "ground_truth_file": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/labels.csv",
                "output_directory": self.output_dir,
                "prompt_characteristics": [
                    "4个详细任务",
                    "复杂验证流程",
                    "详细的few-shot examples",
                    "严格的分类标准",
                    "Temperature=0确保一致性",
                    "专注ghost probing检测",
                    "100视频完整评估",
                    "使用校正后labels.csv"
                ]
            },
            "detailed_results": []
        }
    
    def load_existing_results(self):
        """加载现有的中间结果"""
        import glob
        
        # 查找最新的中间结果文件
        intermediate_files = glob.glob(os.path.join(self.output_dir, "run8_rerun_intermediate_*videos_*.json"))
        
        if intermediate_files:
            latest_file = max(intermediate_files, key=os.path.getmtime)
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # 加载现有结果
                existing_results = existing_data.get('detailed_results', [])
                if existing_results:
                    self.results['detailed_results'] = existing_results
                    self.logger.info(f"📂 加载了 {len(existing_results)} 个现有结果")
                else:
                    self.logger.info("📂 没有找到现有结果")
                    
            except Exception as e:
                self.logger.error(f"❌ 加载现有结果失败: {e}")
        else:
            self.logger.info("📂 没有找到中间结果文件，从头开始")
        
    def extract_frames_from_video(self, video_path, frame_interval=10, frames_per_interval=10):
        """从视频中提取帧 - 与Run 8完全相同"""
        try:
            # 使用moviepy获取视频时长
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.close()
            
            # 如果视频时长小于frame_interval，调整参数
            if duration < frame_interval:
                frame_interval = int(duration)
                frames_per_interval = max(1, int(duration))
            
            # 使用OpenCV提取帧
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0:
                fps = 30  # 默认帧率
            
            frames = []
            frame_times = []
            
            # 计算采样帧的位置
            start_frame = 0
            end_frame = min(int(fps * frame_interval), total_frames - 1)
            
            if end_frame <= start_frame:
                end_frame = start_frame + 1
            
            frame_indices = []
            if frames_per_interval == 1:
                frame_indices = [start_frame + (end_frame - start_frame) // 2]
            else:
                step = (end_frame - start_frame) / (frames_per_interval - 1)
                frame_indices = [int(start_frame + i * step) for i in range(frames_per_interval)]
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    frame_times.append(frame_idx / fps)
            
            cap.release()
            
            return frames, frame_times, duration
            
        except Exception as e:
            self.logger.error(f"视频帧提取失败 {video_path}: {str(e)}")
            return [], [], 0

    def get_paper_batch_prompt_with_fewshot(self, video_id, frame_interval=10, frames_per_interval=10):
        """获取包含Few-shot Examples的Paper_Batch prompt - 与Run 8完全相同"""
        return f'''You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the portion of the video you are considering.

Direction - Please identify the objects in the image based on their position in the image itself. Do not assume your own position within the image. Treat the left side of the image as 'left' and the right side of the image as 'right'. Assume the viewpoint is standing from at the bottom center of the image. Describe whether the objects are on the left or right side of this central point, left is left, and right is right. For example, if there is a car on the left side of the image, state that the car is on the left.

**Task 1: Identify and Predict potential "Ghost Probing(专业术语：鬼探头)" behavior**

"Ghost Probing" includes the following key behaviors:

1) Traditional Ghost Probing: 
   - A person or cyclist suddenly darting out from either left or right side of the car
   - Must emerge from behind a physical obstruction that blocks the driver's view, such as a parked car, a tree, or a wall
   - Directly entering the driver's path with minimal reaction time

2) Vehicle Ghost Probing: 
   - A vehicle suddenly emerging from behind a physical obstruction
   - Examples include: buildings at intersections, parked vehicles, roadside structures, flower beds, a bridge, even a moving car at the front hiding another moving car, etc.
   - Vehicles entering from perpendicular roads that were previously hidden by obstructions

Core Characteristics:
- Presence of a physical obstruction that creates a visual barrier
- Sudden appearance from behind this obstruction with minimal reaction time
- The physical obstruction makes detection impossible until emergence
- Creates an immediate danger or potential collision situation

**Task 2: Character Analysis**
- Identify the number, age, and description of all characters in the images
- Pay attention to pedestrians, cyclists, drivers, and any people visible
- Note their positions, movements, and potential interactions with vehicles

**Task 3: Object Recognition and Tracking**
- Identify and track the movement of key objects throughout the sequence
- Focus on vehicles, pedestrians, traffic signs, and environmental elements
- Note changes in relative positions and distances

**Task 4: Action Prediction**
- Based on the observed scene and identified patterns, predict the most likely next action
- Consider safety implications and emergency scenarios
- Provide specific recommendations for speed control, direction control, and lane management

## Few-shot Examples:

### Example 1: Ghost Probing Detection
Video: images_1_005.avi
Analysis: "The sequence shows a vehicle approaching an intersection with parked cars on the right side. At 8 seconds, a cyclist suddenly emerges from behind a parked black sedan on the right side and enters the driving lane. This is a clear case of 'ghost probing' as the cyclist was completely hidden by the parked vehicle until the moment of emergence, creating an immediate collision risk."
Key Action: "ghost probing"

### Example 2: No Ghost Probing - Clear View
Video: images_1_009.avi  
Analysis: "The sequence shows a pedestrian walking along the roadside. However, the pedestrian is clearly visible throughout the entire sequence without any visual obstructions. While the pedestrian is present, there is no sudden emergence from behind an obstruction, so this does not constitute 'ghost probing'."
Key Action: "pedestrian crossing, no obstruction"

### Example 3: Vehicle Ghost Probing
Video: images_4_002.avi
Analysis: "At 6 seconds, a white vehicle suddenly appears from behind a building at the intersection, entering the main road perpendicular to the camera vehicle's path. The building created a complete visual obstruction until the vehicle emerged, making this a clear case of vehicle 'ghost probing'."
Key Action: "ghost probing"

## Output Format:
You must respond with ONLY a valid JSON object in this exact format:

```json
{{
    "video_id": "{video_id}",
    "segment_id": "full_video",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "{frame_interval}.0s",
    "sentiment": "Positive/Neutral/Negative",
    "scene_theme": "Safe/Cautious/Dangerous",
    "characters": "Detailed description of all people observed",
    "summary": "Comprehensive analysis of the entire video sequence",
    "actions": "Detailed description of all observed actions and movements",
    "key_objects": "List and description of important objects and their movements",
    "key_actions": "ghost probing" OR "no ghost probing" OR specific action description,
    "next_action": {{
        "speed_control": "maintain/accelerate/decelerate/brake/emergency brake",
        "direction_control": "straight/left/right/emergency maneuver",
        "lane_control": "maintain current lane/change to left/change to right"
    }}
}}
```

Audio Transcription: [No audio available for this analysis]

Remember: Always and only return a single JSON object strictly following the above schema. Be incredibly detailed in your analysis, especially for ghost probing detection. Use the examples above as guidance for the level of detail and accuracy expected.'''
    
    def send_azure_openai_request(self, prompt, images):
        """发送Azure OpenAI请求 - 使用Temperature=0"""
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
            "max_tokens": 3000,
            "temperature": 0  # 确保使用Temperature=0
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
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API请求失败: {response.status_code}")
                self.logger.error(f"响应内容: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"API请求异常: {str(e)}")
            return None

    def save_frames(self, frames, video_id, temp_dir):
        """保存帧到临时目录"""
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_filename = f"{video_id}_frame_{i+1}.jpg"
            frame_path = os.path.join(temp_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        return frame_paths

    def analyze_video(self, video_path, video_id):
        """分析单个视频"""
        self.logger.info(f"🎬 开始分析视频: {video_id}")
        
        try:
            # 提取帧
            frames, frame_times, duration = self.extract_frames_from_video(video_path)
            if not frames:
                self.logger.error(f"❌ 无法提取帧: {video_id}")
                return None
            
            # 保存临时帧
            temp_dir = os.path.join(self.output_dir, "frames_temp")
            os.makedirs(temp_dir, exist_ok=True)
            frame_paths = self.save_frames(frames, video_id, temp_dir)
            
            # 获取prompt
            prompt = self.get_paper_batch_prompt_with_fewshot(video_id)
            
            # 发送API请求
            self.logger.info(f"📤 发送API请求: {video_id}")
            response = self.send_azure_openai_request(prompt, frame_paths)
            
            # 清理临时文件
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
            
            if not response:
                self.logger.error(f"❌ API响应为空: {video_id}")
                return None
            
            # 解析响应
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            if not content:
                self.logger.error(f"❌ 响应内容为空: {video_id}")
                return None
            
            # 提取JSON
            try:
                # 尝试直接解析JSON
                if content.strip().startswith('{'):
                    result = json.loads(content.strip())
                else:
                    # 查找JSON代码块
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(1))
                    else:
                        # 查找JSON对象
                        json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group(1))
                        else:
                            raise ValueError("无法找到JSON格式")
                
                # 获取ground truth
                gt_row = self.ground_truth[self.ground_truth['video_id'] == f"{video_id}.avi"]
                if not gt_row.empty:
                    gt_label = gt_row.iloc[0]['ground_truth_label']
                    if 'ghost probing' in str(gt_label).lower():
                        ground_truth = "ghost_probing"
                    else:
                        ground_truth = "none"
                else:
                    ground_truth = "unknown"
                
                # 提取key_actions进行评估
                key_actions = result.get('key_actions', '').lower()
                if 'no ghost probing' in key_actions or 'not ghost probing' in key_actions:
                    prediction = "none"
                elif 'ghost probing' in key_actions:
                    prediction = "ghost_probing"
                else:
                    prediction = "none"
                
                # 评估结果
                if ground_truth == "unknown":
                    evaluation = "UNKNOWN"
                elif ground_truth == prediction:
                    evaluation = "TP" if prediction == "ghost_probing" else "TN"
                else:
                    evaluation = "FP" if prediction == "ghost_probing" else "FN"
                
                self.logger.info(f"✅ 分析完成: {video_id} - {evaluation}")
                
                return {
                    "video_id": f"{video_id}.avi",
                    "ground_truth": ground_truth,
                    "key_actions": result.get('key_actions', ''),
                    "evaluation": evaluation,
                    "raw_result": json.dumps(result, ensure_ascii=False, indent=2)
                }
                
            except Exception as e:
                self.logger.error(f"❌ JSON解析失败 {video_id}: {str(e)}")
                self.logger.error(f"原始内容: {content[:500]}...")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 视频分析异常 {video_id}: {str(e)}")
            return None

    def run_experiment(self):
        """运行完整实验"""
        self.logger.info("🚀 开始Run 8重新运行实验")
        
        # DADA-100视频目录
        video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
        
        # 获取视频列表
        video_files = []
        for i in range(1, 6):  # images_1 到 images_5
            for j in range(1, 100):  # 001 到 099
                if i == 2 and j == 5:  # 跳过缺失的images_2_005
                    continue
                video_name = f"images_{i}_{j:03d}.avi"
                video_path = os.path.join(video_dir, video_name)
                if os.path.exists(video_path):
                    video_files.append((video_path, f"images_{i}_{j:03d}"))
        
        # 添加补充视频
        supplement_video = os.path.join(video_dir, "images_5_055.avi")
        if os.path.exists(supplement_video):
            video_files.append((supplement_video, "images_5_055"))
        
        self.logger.info(f"📊 找到 {len(video_files)} 个视频文件")
        
        # 检查已处理的视频
        processed_videos = set()
        for existing_result in self.results.get("detailed_results", []):
            video_id = existing_result.get("video_id", "").replace(".avi", "")
            processed_videos.add(video_id)
        
        # 批量处理
        total_processed = len(processed_videos)
        successful_results = list(self.results.get("detailed_results", []))
        
        for video_path, video_id in tqdm.tqdm(video_files, desc="处理视频"):
            # 跳过已处理的视频
            if video_id in processed_videos:
                continue
                
            result = self.analyze_video(video_path, video_id)
            if result:
                successful_results.append(result)
                self.results["detailed_results"].append(result)
                total_processed += 1
                
                # 每10个视频保存一次中间结果
                if total_processed % 10 == 0:
                    self.save_intermediate_results(total_processed)
            
            # 避免API限制
            time.sleep(1)
        
        # 保存最终结果
        self.save_final_results(total_processed)
        
        self.logger.info(f"🎯 实验完成！处理了 {total_processed} 个视频")
        return successful_results

    def save_intermediate_results(self, count):
        """保存中间结果"""
        filename = os.path.join(self.output_dir, f"run8_rerun_intermediate_{count}videos_{self.timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def save_final_results(self, count):
        """保存最终结果"""
        filename = os.path.join(self.output_dir, f"run8_rerun_final_results_{self.timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"💾 最终结果已保存: {filename}")

if __name__ == "__main__":
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8_gpt4o_ghost_probing_fewshot_100videos_results/rerun_corrected"
    
    experiment = GPT4oRun8RerunCorrectedLabels(output_dir)
    results = experiment.run_experiment()
    
    print(f"\n🎉 Run 8重新运行完成！")
    print(f"📊 成功处理: {len(results)} 个视频")
    print(f"📁 结果目录: {output_dir}")