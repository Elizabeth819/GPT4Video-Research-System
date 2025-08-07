#!/usr/bin/env python3
"""
Run 9: GPT-4o Ghost Probing Detection with Image Few-shot Learning (100 Videos)
基于Run 8架构，增加图像few-shot examples提升性能
最优配置：Paper_Batch架构 + Text Few-shot + Image Few-shot + Temperature=0
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

class GPT4oRun9ImageFewshot:
    def __init__(self, output_dir, chunk_size=10):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logging()
        self.setup_openai_api()
        self.load_ground_truth()
        self.load_few_shot_images()
        self.initialize_results()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        log_filename = os.path.join(self.output_dir, f"run9_image_fewshot_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Run 9: GPT-4o Ghost Probing Detection with Image Few-shot Learning (100 Videos) 开始")
        
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
        self.logger.info(f"Temperature: 0, Enhanced with Text + Image Few-shot Examples")
        
    def load_ground_truth(self):
        """加载ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
        self.ground_truth = pd.read_csv(gt_path, sep='\t')
        self.logger.info(f"加载ground truth标签: {len(self.ground_truth)}个视频")
        
    def load_few_shot_images(self):
        """加载和编码few-shot示例图片"""
        fsl_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/fsl"
        self.few_shot_images = {}
        
        # 定义示例图片分类
        ghost_probing_examples = [
            "frame_at_26s.jpg", "frame_at_27s.jpg", "frame_at_28s.jpg",
            "frame_at_29s.jpg", "frame_at_30s.jpg", "frame_at_31s.jpg"
        ]
        
        lower_barrier_examples = [
            "lowerbar_1s.jpg", "lowerbar_3s.jpg", "lowerbar_5s.jpg", 
            "lowerbar_7s.jpg", "lowerbar_8s.jpg", "lowerbar_9s.jpg"
        ]
        
        red_truck_examples = ["redtruck-32s.png", "redtruck-33s.png"]
        
        all_examples = ghost_probing_examples + lower_barrier_examples + red_truck_examples
        
        loaded_count = 0
        for img_name in all_examples:
            img_path = os.path.join(fsl_dir, img_name)
            if os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as f:
                        base64_img = base64.b64encode(f.read()).decode('utf-8')
                        self.few_shot_images[img_name] = base64_img
                        loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"加载图片失败 {img_name}: {str(e)}")
        
        # 按类别组织图片
        self.ghost_images = {k: v for k, v in self.few_shot_images.items() if k in ghost_probing_examples}
        self.barrier_images = {k: v for k, v in self.few_shot_images.items() if k in lower_barrier_examples}
        self.truck_images = {k: v for k, v in self.few_shot_images.items() if k in red_truck_examples}
        
        self.logger.info(f"成功加载 {loaded_count} 张few-shot示例图片")
        self.logger.info(f"Ghost Probing序列: {len(self.ghost_images)}张")
        self.logger.info(f"Lower Barrier示例: {len(self.barrier_images)}张")
        self.logger.info(f"Red Truck示例: {len(self.truck_images)}张")
        
    def initialize_results(self):
        """初始化结果结构"""
        self.results = {
            "experiment_info": {
                "run_id": "Run 9",
                "timestamp": self.timestamp,
                "video_count": 100,
                "model": "GPT-4o (Azure)",
                "prompt_version": "Paper_Batch Complex (4-Task) + Text Few-shot + Image Few-shot Examples",
                "temperature": 0,
                "max_tokens": 3000,
                "purpose": "Image Few-shot Learning增强ghost probing检测",
                "output_directory": self.output_dir,
                "few_shot_images_loaded": len(self.few_shot_images),
                "prompt_characteristics": [
                    "4个详细任务",
                    "复杂验证流程",
                    "文本few-shot examples",
                    "图像few-shot examples",
                    "视觉模式识别增强",
                    "严格的分类标准",
                    "Temperature=0确保一致性",
                    "专注ghost probing检测",
                    "100视频完整评估"
                ]
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
    
    def get_paper_batch_prompt_with_image_fewshot(self, video_id, frame_interval=10, frames_per_interval=10):
        """获取包含Text + Image Few-shot Examples的Paper_Batch prompt"""
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

Note: Only those emerging from behind a physical obstruction can be considered as 鬼探头 (ghost probing).

**Task 2: Explain Current Driving Actions**
Analyze the current video frames to extract actions. Describe not only the actions themselves but also provide detailed reasoning for why the vehicle is taking these actions, such as changes in speed and direction. Focus solely on the reasoning for the vehicle's actions, excluding any descriptions of pedestrian behavior. Explain why the driver is driving at a certain speed, making turns, or stopping. Your goal is to provide a comprehensive understanding of the vehicle's behavior based on the visual data. Output in the "actions" field of the JSON format template.

**Task 3: Predict Next Driving Action**
Understand the current road conditions, the driving behavior, and to predict the next driving action. Analyze the video and audio to provide a comprehensive summary of the road conditions, including weather, traffic density, road obstacles, and traffic light if visible. Predict the next driving action based on two dimensions, one is driving speed control, such as accelerating, braking, turning, or stopping, the other one is to predict the next lane control, such as change to left lane, change to right lane, keep left in this lane, keep right in this lane, keep straight. Your summary should help understand not only what is happening at the moment but also what is likely to happen next with logical reasoning. The principle is safety first, so the prediction action should prioritize the driver's safety and secondly the pedestrians' safety. Be incredibly detailed. Output in the "next_action" field of the JSON format template.

**Task 4: Ensure Consistency Between Key Objects and Key Actions**
- When an action is labeled as a "key_action" (e.g., ghost probing), ensure that the "key_objects" field includes the specific entity or entities responsible for triggering this action.

Additional Requirements:
- `key_actions` must strictly adhere to the predefined categories:
    - ghost probing
    - overtaking, specify "left-side overtaking" or "right-side overtaking" when relevant.
    - none (if no dangerous behavior is observed)

- All textual fields must be in English.
- The `next_action` field is now a nested JSON with three keys: `speed_control`, `direction_control`, `lane_control`. Each must choose one value from their respective sets.

**Text Few-shot Examples:**

Example 1 - Ghost Probing Detection:
{{
    "video_id": "example_ghost_probing",
    "segment_id": "segment_000",
    "Start_Timestamp": "2.0s",
    "End_Timestamp": "8.0s",
    "sentiment": "Negative",
    "scene_theme": "Dangerous",
    "characters": "Male pedestrian in dark clothing, approximately 25 years old",
    "summary": "In this segment, the vehicle is driving on an urban road with parked vehicles on both sides. A male pedestrian wearing dark clothing suddenly emerges from behind a parked white truck on the right side and steps directly into the vehicle's path. The pedestrian appears from behind the obstruction with minimal warning time, creating a dangerous ghost probing situation.",
    "actions": "The self-driving vehicle is maintaining steady speed when the pedestrian suddenly appears. The vehicle immediately begins rapid deceleration in response to the unexpected pedestrian emergence from behind the parked truck. The driver takes evasive action to avoid collision.",
    "key_objects": "1) Right side: A male pedestrian, approximately 25 years old wearing dark clothing, 3 meters away, suddenly emerging from behind a parked white truck and stepping into the vehicle's path. 2) Right side: A white truck, approximately 5 meters away, parked and creating visual obstruction that hides the pedestrian until emergence.",
    "key_actions": "ghost probing",
    "next_action": {{
        "speed_control": "rapid deceleration",
        "direction_control": "keep direction",
        "lane_control": "maintain current lane"
    }}
}}

Example 2 - No Dangerous Behavior:
{{
    "video_id": "example_normal_driving",
    "segment_id": "segment_000", 
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "10.0s",
    "sentiment": "Neutral",
    "scene_theme": "Routine",
    "characters": "None visible in immediate vicinity",
    "summary": "In this segment, the vehicle is driving on a clear rural road during daytime. The road ahead is clear with good visibility. There are no pedestrians, cyclists, or other vehicles creating any immediate safety concerns. The driving conditions are calm and routine.",
    "actions": "The self-driving vehicle maintains consistent speed and direction on the clear road. No sudden changes in speed or direction are required as there are no obstacles or safety concerns present. The vehicle continues with normal driving behavior.",
    "key_objects": "None requiring immediate attention",
    "key_actions": "none",
    "next_action": {{
        "speed_control": "maintain speed",
        "direction_control": "keep direction", 
        "lane_control": "maintain current lane"
    }}
}}

Example 3 - Vehicle Ghost Probing:
{{
    "video_id": "example_vehicle_ghost",
    "segment_id": "segment_000",
    "Start_Timestamp": "5.0s", 
    "End_Timestamp": "12.0s",
    "sentiment": "Negative",
    "scene_theme": "Dramatic",
    "characters": "Driver of red sedan",
    "summary": "In this segment, the vehicle approaches an intersection with buildings on both sides creating limited visibility. A red sedan suddenly emerges from behind a building on the left side, entering from a perpendicular side street directly into the main road where the self-vehicle is traveling. The sedan was completely hidden by the building structure until it emerged into the intersection.",
    "actions": "The self-driving vehicle is traveling at normal speed when the red sedan suddenly appears from the left side intersection. The vehicle immediately initiates emergency braking and slight steering adjustment to avoid collision with the suddenly appearing vehicle.",
    "key_objects": "1) Left side: A red sedan, approximately 4 meters away, suddenly emerging from behind a building at the intersection and entering the main road. 2) Left side: A large building, approximately 10 meters away, creating visual obstruction that completely hides approaching vehicles until they emerge into the intersection.",
    "key_actions": "ghost probing",
    "next_action": {{
        "speed_control": "rapid deceleration",
        "direction_control": "slight right adjustment",
        "lane_control": "maintain current lane"
    }}
}}

**Visual Few-shot Examples:**

The following visual examples demonstrate actual ghost probing scenarios. Study these images carefully to recognize visual patterns:

VISUAL EXAMPLE SET 1 - Sequential Ghost Probing Emergence:
These images show a time sequence where a pedestrian emerges from behind an obstruction:
- Notice the physical obstruction (parked vehicle, barrier, etc.)
- Observe how the pedestrian/object is initially hidden
- See the sudden emergence creating immediate danger
- Note the minimal reaction time available

VISUAL EXAMPLE SET 2 - Lower Barrier Ghost Probing:
These images demonstrate ghost probing with lower visual barriers:
- Physical obstructions at lower levels (flower beds, low walls, barriers)
- Pedestrians or cyclists emerging from behind these barriers
- Critical timing where detection is nearly impossible until emergence

VISUAL EXAMPLE SET 3 - Vehicle Ghost Probing:
These images show vehicles emerging from behind obstructions:
- Red truck examples showing vehicle-to-vehicle ghost probing
- Notice how the obstruction completely hides the approaching vehicle
- Observe the critical moment of emergence into the driving path

Your response should be a valid JSON object with the following EXACT structure:
{{
    "video_id": "{video_id}",
    "segment_id": "full_video",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "{frame_interval}.0s",
    "sentiment": "Positive/Negative/Neutral",
    "scene_theme": "Dramatic/Routine/Dangerous/Safe",
    "characters": "brief description of people in the scene with specific details (age, gender, clothing, transportation)",
    "summary": "comprehensive summary of the scene and what happens with incredible detail",
    "actions": "actions taken by the vehicle and driver responses with detailed reasoning",
    "key_objects": "numbered list: 1) Position: object description, distance, behavior impact 2) Position: object description, distance, behavior impact",
    "key_actions": "ghost probing/left-side overtaking/right-side overtaking/none",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}

Audio Transcription: [No audio available for this analysis]

Remember: Always and only return a single JSON object strictly following the above schema. Be incredibly detailed in your analysis, especially for ghost probing detection. Use both the text examples and the visual examples above as guidance for recognizing ghost probing patterns. The visual examples should help you identify key visual cues that indicate potential ghost probing situations.'''
    
    def send_azure_openai_request_with_image_fewshot(self, prompt, video_frames):
        """发送Azure OpenAI请求 - 包含图像few-shot examples"""
        # 编码视频帧
        encoded_video_frames = []
        for frame_path in video_frames:
            try:
                with open(frame_path, 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    encoded_video_frames.append(encoded_string)
            except Exception as e:
                self.logger.error(f"视频帧编码失败 {frame_path}: {str(e)}")
                continue
        
        if not encoded_video_frames:
            return None
            
        # 构建内容数组
        content = [{"type": "text", "text": prompt}]
        
        # 添加图像few-shot examples
        content.append({"type": "text", "text": "\n**VISUAL FEW-SHOT EXAMPLES:**\n"})
        
        # 添加Ghost Probing序列示例
        if self.ghost_images:
            content.append({"type": "text", "text": "Ghost Probing Time Sequence Examples:"})
            for img_name, base64_img in sorted(self.ghost_images.items()):
                content.append({"type": "text", "text": f"Example frame: {img_name}"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                })
        
        # 添加Lower Barrier示例
        if self.barrier_images:
            content.append({"type": "text", "text": "Lower Barrier Ghost Probing Examples:"})
            for img_name, base64_img in sorted(self.barrier_images.items()):
                content.append({"type": "text", "text": f"Example frame: {img_name}"})
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                })
        
        # 添加Vehicle Ghost Probing示例
        if self.truck_images:
            content.append({"type": "text", "text": "Vehicle Ghost Probing Examples:"})
            for img_name, base64_img in sorted(self.truck_images.items()):
                content.append({"type": "text", "text": f"Example frame: {img_name}"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })
        
        # 添加当前视频帧进行分析
        content.append({"type": "text", "text": "\n**NOW ANALYZE THESE VIDEO FRAMES:**\n"})
        
        for i, encoded_frame in enumerate(encoded_video_frames):
            content.append({"type": "text", "text": f"Video Frame {i+1}:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"}
            })
        
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 3000,
            "temperature": 0
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
                timeout=90  # 增加超时时间由于图片更多
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            self.logger.error(f"API调用失败: {str(e)}")
            return None
    
    def analyze_with_gpt4o(self, video_path, video_id):
        """使用GPT-4o分析视频（增强图像few-shot）"""
        try:
            # 提取帧
            frames = self.extract_frames_from_video(video_path)
            if not frames:
                return None
            
            # 生成prompt
            prompt = self.get_paper_batch_prompt_with_image_fewshot(video_id)
            
            # 发送API请求（包含图像few-shot）
            result = self.send_azure_openai_request_with_image_fewshot(prompt, frames)
            
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
            key_actions_match = re.search(r'"key_actions":\s*"([^"]*)"', result_text)
            if key_actions_match:
                return key_actions_match.group(1).lower()
            return result_text.lower()
    
    def evaluate_result(self, video_id, key_actions, ground_truth_label):
        """评估结果"""
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
    
    def run_experiment(self):
        """运行Run 9图像few-shot增强实验"""
        # 从ground truth文件中获取完整的100个视频列表
        test_videos = self.ground_truth['video_id'].tolist()
        
        self.logger.info(f"开始Run 9实验，处理 {len(test_videos)} 个视频")
        self.logger.info(f"图像few-shot加载: {len(self.few_shot_images)} 张")
        
        start_time = time.time()
        
        for i, video_id in enumerate(tqdm.tqdm(test_videos, desc="处理视频")):
            try:
                self.logger.info(f"处理视频 {i+1}/100: {video_id}")
                
                # 视频路径
                video_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/{video_id}"
                if not os.path.exists(video_path):
                    self.logger.warning(f"视频不存在: {video_path}")
                    continue
                
                # 获取ground truth
                gt_row = self.ground_truth[self.ground_truth['video_id'] == video_id]
                if gt_row.empty:
                    self.logger.warning(f"未找到ground truth: {video_id}")
                    continue
                
                ground_truth_label = gt_row.iloc[0]['ground_truth_label']
                
                # 分析视频（使用图像few-shot增强）
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
                
                # 每5个视频保存一次中间结果
                if (i + 1) % 5 == 0:
                    self.save_intermediate_results(i + 1)
                
            except Exception as e:
                self.logger.error(f"处理视频失败 {video_id}: {str(e)}")
                # 记录错误结果
                error_entry = {
                    "video_id": video_id,
                    "ground_truth": ground_truth_label if 'ground_truth_label' in locals() else "unknown",
                    "key_actions": "",
                    "evaluation": "ERROR",
                    "raw_result": f"处理错误: {str(e)}"
                }
                self.results["detailed_results"].append(error_entry)
                continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        self.logger.info(f"Run 9实验完成，总耗时: {total_time/60:.1f} 分钟")
        
        # 保存最终结果
        self.save_final_results()
        self.generate_performance_metrics()
        
    def save_intermediate_results(self, processed_count):
        """保存中间结果"""
        intermediate_file = os.path.join(self.output_dir, f"run9_intermediate_{processed_count}videos_{self.timestamp}.json")
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"中间结果已保存: {intermediate_file}")
    
    def save_final_results(self):
        """保存最终结果"""
        final_file = os.path.join(self.output_dir, f"run9_final_results_{self.timestamp}.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"最终结果已保存: {final_file}")
    
    def generate_performance_metrics(self):
        """生成性能指标"""
        from collections import Counter
        
        evaluations = [r['evaluation'] for r in self.results["detailed_results"]]
        eval_counts = Counter(evaluations)
        
        tp = eval_counts.get('TP', 0)
        fp = eval_counts.get('FP', 0)
        tn = eval_counts.get('TN', 0)
        fn = eval_counts.get('FN', 0)
        errors = eval_counts.get('ERROR', 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "errors": errors,
            "total_videos": len(self.results["detailed_results"]),
            "few_shot_images_used": len(self.few_shot_images)
        }
        
        self.logger.info("=== Run 9 Performance Metrics (Image Few-shot Enhanced) ===")
        self.logger.info(f"精确度: {precision:.3f} ({precision*100:.1f}%)")
        self.logger.info(f"召回率: {recall:.3f} ({recall*100:.1f}%)")
        self.logger.info(f"F1分数: {f1:.3f} ({f1*100:.1f}%)")
        self.logger.info(f"准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
        self.logger.info(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, ERROR: {errors}")
        self.logger.info(f"使用图像few-shot示例: {len(self.few_shot_images)} 张")
        
        # 保存指标
        metrics_file = os.path.join(self.output_dir, f"run9_metrics_{self.timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 创建输出目录
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run9_gpt4o_ghost_probing_image_fewshot"
    
    # 运行实验
    experiment = GPT4oRun9ImageFewshot(output_dir)
    experiment.run_experiment()
    
    print("🎯 Run 9: GPT-4o Ghost Probing Detection with Image Few-shot Learning (100 Videos) 实验完成!")
    print(f"📁 结果保存在: {output_dir}")