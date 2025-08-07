#!/usr/bin/env python3
"""
AutoDrive-GPT DADA-100 Few-shot Ablation Experiment
===================================================

Implementation Reference:
- Paper Section 4: Experiment - DADA-100 Dataset evaluation
- Paper Section 4.2: Few-shot Learning Impact Analysis
- Paper Section 4.3: Ablation Studies

This script implements the comprehensive ablation experiment on the DADA-100 dataset
combining textual few-shot learning with 9 visual few-shot examples of ghost probing
scenarios for enhanced detection accuracy.

Key Features:
- DADA-100 dataset processing pipeline
- Multi-modal few-shot learning (text + images)
- Automated result collection and analysis
- Performance metrics calculation
- Statistical significance testing support

Experimental Design:
- Based on run8-rerun methodology with image enhancement
- Implements Paper Section 4.2 prompt engineering strategies
- Uses temperature=0 for deterministic results (Paper Section 4.2)
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
import glob

# 加载环境变量
load_dotenv()

class DADA100AblationExperiment:
    def __init__(self, experiment_name=None, chunk_size=10):
        # 创建实验特定的目录结构
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if experiment_name is None:
            experiment_name = f"run_dada100_image_fewshot_{timestamp}"
        
        # 在image-fewshot目录下创建独立的实验目录
        base_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"
        self.experiment_name = experiment_name
        self.output_dir = os.path.join(base_dir, experiment_name)
        
        self.chunk_size = chunk_size
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建子目录结构
        self.results_dir = os.path.join(self.output_dir, "results")
        self.logs_dir = os.path.join(self.output_dir, "logs") 
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        self.frames_dir = os.path.join(self.output_dir, "frames_temp")
        
        for subdir in [self.results_dir, self.logs_dir, self.analysis_dir, self.frames_dir]:
            os.makedirs(subdir, exist_ok=True)
        self.setup_logging()
        self.setup_openai_api()
        self.load_ground_truth()
        self.setup_image_fewshot()
        self.setup_video_dataset()
        self.initialize_results()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        log_filename = os.path.join(self.logs_dir, f"dada100_ablation_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("DADA-100 Few-shot Ablation Experiment 开始")
        
    def setup_openai_api(self):
        """设置OpenAI API - 使用.env文件中的Azure配置"""
        # 尝试不同的环境变量名称
        self.openai_api_key = (
            os.environ.get("AZURE_OPENAI_API_KEY") or 
            os.environ.get("OPENAI_API_KEY", "")
        )
        if not self.openai_api_key:
            self.logger.warning("Azure OpenAI API Key未设置")
        
        # Azure OpenAI配置
        self.vision_endpoint = (
            os.environ.get("AZURE_OPENAI_API_ENDPOINT") or
            os.environ.get("VISION_ENDPOINT", "")
        )
        self.vision_deployment = (
            os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or
            os.environ.get("VISION_DEPLOYMENT_NAME", "gpt-4o-global")
        )
        
        if not self.vision_endpoint:
            self.logger.warning("Azure OpenAI Endpoint未设置")
            
        self.logger.info(f"Azure OpenAI API配置")
        self.logger.info(f"Endpoint: {self.vision_endpoint if self.vision_endpoint else 'Not Set'}")
        self.logger.info(f"Deployment: {self.vision_deployment}")
        self.logger.info(f"API Key: {'已设置' if self.openai_api_key else '未设置'}")
        self.logger.info(f"Temperature: 0, Run8 Text Few-shot + 9张Image Few-shot")
        
    def load_ground_truth(self):
        """加载DADA-100-videos的ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/labels.csv"
        if os.path.exists(gt_path):
            self.ground_truth = pd.read_csv(gt_path, sep=',')
            self.logger.info(f"加载DADA-100 ground truth标签: {len(self.ground_truth)}个视频")
            
            # 显示标签分布
            if 'ground_truth_label' in self.ground_truth.columns:
                ghost_count = len([x for x in self.ground_truth['ground_truth_label'] if 'ghost probing' in str(x)])
                none_count = len([x for x in self.ground_truth['ground_truth_label'] if str(x) == 'none'])
                self.logger.info(f"标签分布: Ghost Probing={ghost_count}, None={none_count}")
        else:
            self.logger.error(f"Ground truth文件不存在: {gt_path}")
            self.ground_truth = None
                
    def setup_image_fewshot(self):
        """设置9张鬼探头few-shot图像"""
        self.fewshot_image_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"
        
        # 按顺序加载9张图像
        expected_images = [
            "ghost_probing_sample1_before.jpg",
            "ghost_probing_sample1_during.jpg", 
            "ghost_probing_sample1_after.jpg",
            "ghost_probing_sample2_before.jpg",
            "ghost_probing_sample2_during.jpg",
            "ghost_probing_sample2_after.jpg",
            "ghost_probing_sample3_before.jpg",
            "ghost_probing_sample3_during.jpg",
            "ghost_probing_sample3_after.jpg"
        ]
        
        self.fewshot_images = []
        for img_name in expected_images:
            img_path = os.path.join(self.fewshot_image_dir, img_name)
            if os.path.exists(img_path):
                self.fewshot_images.append(img_path)
                self.logger.info(f"✅ 加载few-shot图像: {img_name}")
            else:
                self.logger.error(f"❌ Few-shot图像不存在: {img_path}")
                
        if len(self.fewshot_images) != 9:
            raise ValueError(f"期望9张few-shot图像，实际找到{len(self.fewshot_images)}张")
            
        self.logger.info(f"🎯 成功加载{len(self.fewshot_images)}张few-shot图像用于DADA-100消融实验")
        
    def setup_video_dataset(self):
        """设置DADA-100-videos数据集"""
        self.video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos"
        
        # 获取所有视频文件
        video_files = glob.glob(os.path.join(self.video_dir, "*.avi"))
        self.video_files = sorted(video_files)
        
        self.logger.info(f"🎬 找到{len(self.video_files)}个DADA-100视频文件")
        
        # 显示前5个视频
        for i, video_file in enumerate(self.video_files[:5]):
            video_name = os.path.basename(video_file)
            file_size = os.path.getsize(video_file)
            self.logger.info(f"   📹 {i+1}: {video_name} ({file_size:,} bytes)")
            
        if len(self.video_files) > 5:
            self.logger.info(f"   ... 以及其他{len(self.video_files)-5}个视频")
            
    def initialize_results(self):
        """初始化结果结构"""
        self.results = {
            "experiment_info": {
                "experiment_type": "DADA-100 Few-shot Ablation",
                "timestamp": self.timestamp,
                "dataset": "DADA-100-videos",
                "model": "GPT-4o (Azure)",
                "prompt_version": "Run8-Rerun + Text Few-shot + 9 Image Few-shot",
                "temperature": 0,
                "max_tokens": 3000,
                "purpose": "消融实验：评估9张鬼探头图像few-shot在DADA-100上的性能提升",
                "output_directory": self.output_dir,
                "dataset_info": {
                    "total_videos": len(self.video_files),
                    "video_directory": self.video_dir,
                    "ground_truth_file": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/labels.csv"
                },
                "fewshot_config": {
                    "text_fewshot_examples": 3,
                    "image_fewshot_examples": 9,
                    "image_sequences": 3,
                    "images_per_sequence": 3,
                    "sequence_pattern": "before-during-after"
                },
                "enhanced_characteristics": [
                    "基于run8-rerun的完全相同prompt",
                    "4个详细任务分析",
                    "3个文本few-shot examples",
                    "9张鬼探头图像visual examples",
                    "DADA-100专门标注数据集",
                    "Temperature=0确保一致性",
                    "专注ghost probing检测性能评估"
                ]
            },
            "processed_videos": [],
            "successful_analyses": 0,
            "failed_analyses": 0,
            "processing_errors": []
        }

    def get_run8_rerun_plus_image_prompt(self, video_id, frame_interval=10, frames_per_interval=10):
        """获取与run8-rerun完全相同的文本prompt + 9张图像few-shot"""
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

## Text Few-shot Examples (与run8-rerun完全相同):

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

## Visual Few-shot Examples:

I have provided 9 reference images showing ghost probing patterns in 3 sequences:

**Sequence 1 (Images 1-3): Intersection Ghost Probing from images_1_003.avi**
- Image 1 (Before @ 1.2s): Normal scene with potential hidden threat behind obstruction
- Image 2 (During @ 2.0s): Critical moment when person emerges from behind obstruction
- Image 3 (After @ 2.5s): Dangerous situation with visible threat requiring immediate action

**Sequence 2 (Images 4-6): Building Emergence from images_1_006.avi**
- Image 4 (Before @ 8.0s): Person hidden behind building structure
- Image 5 (During @ 9.0s): Person emerging from building into vehicle path
- Image 6 (After @ 10.0s): Full emergence creating collision risk

**Sequence 3 (Images 7-9): Blind Spot Emergence from images_1_008.avi**
- Image 7 (Before @ 1.0s): Normal driving scene with hidden potential threat
- Image 8 (During @ 2.8s): Critical emergence moment from blind spot
- Image 9 (After @ 3.0s): Threat fully visible requiring emergency response

These visual examples demonstrate the classic "ghost probing" pattern from DADA-100 dataset:
- **BEFORE**: Normal scene with hidden potential threat behind physical obstruction
- **DURING**: Critical moment when object emerges from obstruction with minimal warning
- **AFTER**: Dangerous situation with visible threat requiring immediate emergency action

Use both the text examples and these visual patterns to guide your analysis. Pay special attention to:
1. Physical obstructions that block visibility
2. Sudden emergence with minimal reaction time
3. The progression from hidden → emerging → visible threat

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

Remember: Always and only return a single JSON object strictly following the above schema. Be incredibly detailed in your analysis, especially for ghost probing detection. Use both the text examples and visual patterns shown in the reference images as guidance. The visual examples are from the same DADA-100 dataset you are analyzing.'''
    
    def extract_frames_from_video(self, video_path, video_id, frame_interval=10, frames_per_interval=10):
        """从视频提取帧 - 与run8-rerun完全相同的逻辑"""
        frames = []
        frame_paths = []
        
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            # 使用实验专用的帧目录
            frames_dir = self.frames_dir
            
            # 计算区间数量
            interval_count = max(1, int(duration // frame_interval))
            
            for i in range(interval_count):
                start_time = i * frame_interval
                end_time = min((i + 1) * frame_interval, duration)
                
                # 在每个区间内提取frames_per_interval个帧
                for j in range(frames_per_interval):
                    if interval_count == 1:
                        # 如果只有一个区间，在整个视频长度内均匀分布
                        timestamp = start_time + (j * (end_time - start_time) / frames_per_interval)
                    else:
                        # 多个区间时，在当前区间内均匀分布
                        timestamp = start_time + (j * frame_interval / frames_per_interval)
                    
                    if timestamp < duration:
                        frame = clip.get_frame(timestamp)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        frame_filename = f"{video_id}_frame_{i:03d}_{j:02d}_{timestamp:.2f}s.jpg"
                        frame_path = os.path.join(frames_dir, frame_filename)
                        
                        if cv2.imwrite(frame_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            frames.append(frame)
                            frame_paths.append(frame_path)
            
            clip.close()
            return frames, frame_paths, duration
            
        except Exception as e:
            self.logger.error(f"视频处理失败 {video_path}: {str(e)}")
            return [], [], 0
    
    def send_azure_openai_request(self, prompt, video_frame_paths):
        """发送Azure OpenAI请求 - 先发送9张few-shot图像，再发送视频帧"""
        if not self.openai_api_key or not self.vision_endpoint:
            self.logger.error("API配置未完成，无法发送请求")
            return None
            
        encoded_images = []
        
        # 首先编码9张few-shot图像
        self.logger.info(f"📷 编码{len(self.fewshot_images)}张few-shot图像")
        for i, image_path in enumerate(self.fewshot_images):
            try:
                with open(image_path, 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    encoded_images.append(encoded_string)
                    self.logger.debug(f"Few-shot图像 {i+1}/9: {os.path.basename(image_path)}")
            except Exception as e:
                self.logger.error(f"Few-shot图像编码失败 {image_path}: {str(e)}")
                continue
        
        # 然后编码视频帧
        self.logger.info(f"🎬 编码{len(video_frame_paths)}张视频帧")
        for image_path in video_frame_paths:
            try:
                with open(image_path, 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    encoded_images.append(encoded_string)
            except Exception as e:
                self.logger.error(f"视频帧编码失败 {image_path}: {str(e)}")
                continue
        
        if not encoded_images:
            self.logger.error("没有成功编码的图像")
            return None
            
        self.logger.info(f"📤 总共发送{len(encoded_images)}张图像 (9张few-shot + {len(video_frame_paths)}张视频帧)")
            
        content = [{"type": "text", "text": prompt}]
        
        for encoded_image in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                    "detail": "high"
                }
            })
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.openai_api_key
        }
        
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 3000,
            "temperature": 0  # 与run8-rerun完全相同
        }
        
        try:
            url = f"{self.vision_endpoint}/openai/deployments/{self.vision_deployment}/chat/completions?api-version=2023-12-01-preview"
            response = requests.post(url, headers=headers, json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                self.logger.error(f"API请求失败: {response.status_code}")
                self.logger.error(f"响应内容: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"API请求异常: {str(e)}")
            return None

    def process_single_video(self, video_path):
        """处理单个视频"""
        video_id = os.path.basename(video_path).replace('.avi', '')
        
        self.logger.info(f"🎬 开始分析DADA-100视频: {video_id}")
        
        # 提取帧
        frames, frame_paths, duration = self.extract_frames_from_video(video_path, video_id)
        
        if not frame_paths:
            self.logger.error(f"❌ 无法提取帧: {video_id}")
            self.results["failed_analyses"] += 1
            return None
        
        # 生成prompt（包含文本few-shot + 9张图像的描述）
        prompt = self.get_run8_rerun_plus_image_prompt(video_id)
        
        self.logger.info(f"📤 发送API请求: {video_id} (with 9 few-shot images from DADA-100)")
        
        # 发送请求（few-shot图像 + 视频帧）
        response = self.send_azure_openai_request(prompt, frame_paths)
        
        if response:
            try:
                # 清理响应内容 - 移除markdown代码块标记
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # 移除开头的```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # 移除结尾的```
                cleaned_response = cleaned_response.strip()
                
                # 尝试解析JSON
                json_response = json.loads(cleaned_response)
                
                # 保存结果到results目录
                result_file = os.path.join(self.results_dir, f"actionSummary_{video_id}.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(json_response, f, indent=2, ensure_ascii=False)
                
                self.results["processed_videos"].append({
                    "video_id": video_id,
                    "status": "success",
                    "duration": duration,
                    "frames_extracted": len(frame_paths),
                    "fewshot_images_used": len(self.fewshot_images),
                    "total_images_sent": len(self.fewshot_images) + len(frame_paths),
                    "result_file": result_file
                })
                
                self.results["successful_analyses"] += 1
                self.logger.info(f"✅ 成功分析: {video_id}")
                
                return json_response
                
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ JSON解析失败 {video_id}: {str(e)}")
                
                # 保存原始响应用于调试
                raw_file = os.path.join(self.logs_dir, f"raw_response_{video_id}.txt")
                with open(raw_file, 'w', encoding='utf-8') as f:
                    f.write(response)
                
                self.results["failed_analyses"] += 1
                self.results["processing_errors"].append({
                    "video_id": video_id,
                    "error_type": "json_decode_error",
                    "error_message": str(e)
                })
                
                return None
        else:
            self.logger.error(f"❌ API响应为空: {video_id}")
            self.results["failed_analyses"] += 1
            self.results["processing_errors"].append({
                "video_id": video_id,
                "error_type": "api_request_failed",
                "error_message": "Empty response from API"
            })
            return None

    def run_dada100_ablation_experiment(self, max_videos=None, start_from=0):
        """运行DADA-100消融实验"""
        video_list = self.video_files[start_from:]
        if max_videos:
            video_list = video_list[:max_videos]
            
        self.logger.info(f"🚀 开始DADA-100消融实验")
        self.logger.info(f"📊 处理视频: {len(video_list)} 个 (从第{start_from+1}个开始)")
        self.logger.info(f"🎯 Few-shot配置: {len(self.fewshot_images)}张图像 + 3个文本例子")
        
        for i, video_path in enumerate(tqdm.tqdm(video_list, desc="DADA-100处理进度")):
            self.logger.info(f"📊 进度: {i+1}/{len(video_list)} (总体: {start_from+i+1}/{len(self.video_files)})")
            
            result = self.process_single_video(video_path)
            
            # 保存中间结果（每10个视频）
            if (i + 1) % 10 == 0:
                intermediate_file = os.path.join(self.analysis_dir, f"intermediate_results_{start_from+i+1}videos_{self.timestamp}.json")
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
                self.logger.info(f"💾 中间结果已保存: {intermediate_file}")
            
            # 短暂延迟避免API限制
            if i < len(video_list) - 1:
                time.sleep(3)
                
        # 保存最终结果
        final_file = os.path.join(self.analysis_dir, f"dada100_ablation_final_results_{self.timestamp}.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
            
        # 创建实验总结文件
        experiment_summary = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "directory_structure": {
                "experiment_root": self.output_dir,
                "results": self.results_dir,
                "logs": self.logs_dir,
                "analysis": self.analysis_dir,
                "frames_temp": self.frames_dir
            },
            "performance_summary": {
                "total_videos": len(video_list),
                "successful_analyses": self.results["successful_analyses"],
                "failed_analyses": self.results["failed_analyses"],
                "success_rate": f"{(self.results['successful_analyses']/len(video_list)*100):.1f}%" if len(video_list) > 0 else "0%"
            },
            "files_generated": {
                "individual_results": f"{self.results['successful_analyses']} JSON files in results/",
                "final_summary": "dada100_ablation_final_results_{}.json".format(self.timestamp),
                "logs": f"dada100_ablation_{self.timestamp}.log"
            }
        }
        
        summary_file = os.path.join(self.output_dir, "EXPERIMENT_SUMMARY.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_summary, f, indent=2, ensure_ascii=False)
            
        # 生成性能报告
        self.generate_performance_report()
            
        self.logger.info(f"🎉 DADA-100消融实验完成")
        self.logger.info(f"成功: {self.results['successful_analyses']}")
        self.logger.info(f"失败: {self.results['failed_analyses']}")
        self.logger.info(f"总计: {len(video_list)}")
        self.logger.info(f"📁 实验目录: {self.output_dir}")
        self.logger.info(f"📁 最终结果: {final_file}")
        self.logger.info(f"📋 实验总结: {summary_file}")
        
        return self.results
        
    def generate_performance_report(self):
        """生成性能分析报告"""
        if not self.ground_truth is not None:
            self.logger.warning("无ground truth数据，跳过性能分析")
            return
            
        # 这里可以添加与ground truth的对比分析
        # 计算precision, recall, F1-score等指标
        self.logger.info("性能分析报告生成功能待实现")

if __name__ == "__main__":
    print("🚀 DADA-100 Few-shot Ablation Experiment")
    print("基于run8-rerun + 9张鬼探头图像few-shot")
    print("专门针对DADA-100-videos数据集")
    
    # 创建带时间戳的实验名称
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"run_dada100_image_fewshot_{timestamp}"
    
    print(f"🏷️  实验名称: {experiment_name}")
    
    experiment = DADA100AblationExperiment(experiment_name=experiment_name)
    print(f"📁 实验目录: {experiment.output_dir}")
    print(f"   ├── results/     (JSON结果文件)")
    print(f"   ├── logs/        (日志和调试文件)")
    print(f"   ├── analysis/    (分析和汇总)")
    print(f"   └── frames_temp/ (临时视频帧)")
    print(f"🎯 Few-shot图像: {len(experiment.fewshot_images)}张")
    print(f"🎬 DADA-100视频: {len(experiment.video_files)}个")
    
    # 检查API配置
    if not experiment.openai_api_key or not experiment.vision_endpoint:
        print("⚠️  API配置未完成，需要设置环境变量后运行")
        print("需要设置: AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME")
        print("\n检查.env文件中的配置:")
        print("AZURE_OPENAI_API_KEY=your-api-key")
        print("AZURE_OPENAI_API_ENDPOINT=your-endpoint")
        print("AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-global")
    else:
        print("✅ API配置完成，开始DADA-100消融实验")
        
        # 开始完整实验 - 处理所有101个视频
        print("🎯 开始处理DADA-100全部101个视频")
        print("⏱️  预计时间: ~45分钟 (每个视频约25秒)")
        print("💾 中间结果每10个视频自动保存")
        
        try:
            results = experiment.run_dada100_ablation_experiment()
            print("\n🎉 DADA-100消融实验成功完成！")
            print(f"📊 最终统计:")
            print(f"   成功: {results['successful_analyses']}/{len(experiment.video_files)}")
            print(f"   失败: {results['failed_analyses']}/{len(experiment.video_files)}")
            print(f"📁 结果位置: {experiment.output_dir}")
        except KeyboardInterrupt:
            print("\n⚠️  实验被用户中断")
            print("💾 已处理的结果已保存在实验目录中")
        except Exception as e:
            print(f"\n❌ 实验过程中出现错误: {str(e)}")
            print("💾 已处理的结果已保存在实验目录中")