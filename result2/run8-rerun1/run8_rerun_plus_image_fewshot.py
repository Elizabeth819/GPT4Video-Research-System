#!/usr/bin/env python3
"""
Run8-Rerun + Image Few-shot Experiment
基于run8-rerun的完全相同prompt + 9张鬼探头图像作为visual few-shot
文本few-shot + 图像few-shot的组合实验
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

class Run8RerunPlusImageFewshot:
    def __init__(self, output_dir=None, chunk_size=10):
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-rerun1/run8_rerun_plus_image_results_{timestamp}"
        else:
            self.output_dir = output_dir
            
        self.chunk_size = chunk_size
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logging()
        self.setup_openai_api()
        self.load_ground_truth()
        self.setup_image_fewshot()
        self.initialize_results()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        log_filename = os.path.join(self.output_dir, f"run8_rerun_plus_image_fewshot_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Run8-Rerun + Image Few-shot Experiment 开始")
        
    def setup_openai_api(self):
        """设置OpenAI API - 与run8-rerun完全相同的配置"""
        self.openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY未设置")
        
        # Azure OpenAI配置
        self.vision_endpoint = os.environ.get("AZURE_OPENAI_API_ENDPOINT", "")
        self.vision_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-global")
        
        if not self.vision_endpoint:
            raise ValueError("AZURE_OPENAI_API_ENDPOINT未设置")
            
        self.logger.info(f"Azure OpenAI API配置成功")
        self.logger.info(f"Endpoint: {self.vision_endpoint}")
        self.logger.info(f"Deployment: {self.vision_deployment}")
        self.logger.info(f"Temperature: 0, Run8 Text Few-shot + 9张Image Few-shot")
        
    def load_ground_truth(self):
        """加载ground truth标签 - 使用与run8-rerun相同的corrected labels"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/DADA-200-videos/labels.csv"
        if os.path.exists(gt_path):
            self.ground_truth = pd.read_csv(gt_path, sep='\t')
            self.logger.info(f"加载校正后的ground truth标签: {len(self.ground_truth)}个视频")
        else:
            # fallback to original ground truth
            gt_path_fallback = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
            if os.path.exists(gt_path_fallback):
                self.ground_truth = pd.read_csv(gt_path_fallback, sep='\t')
                self.logger.info(f"加载原始ground truth标签: {len(self.ground_truth)}个视频")
            else:
                self.logger.warning("未找到ground truth文件")
                self.ground_truth = None
                
    def setup_image_fewshot(self):
        """设置9张鬼探头few-shot图像"""
        self.fewshot_image_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8-rerun1"
        
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
            
        self.logger.info(f"🎯 成功加载{len(self.fewshot_images)}张few-shot图像用于消融实验")
        
    def initialize_results(self):
        """初始化结果结构"""
        self.results = {
            "experiment_info": {
                "experiment_type": "Run8-Rerun + Image Few-shot",
                "timestamp": self.timestamp,
                "model": "GPT-4o (Azure)",
                "prompt_version": "Paper_Batch Complex (4-Task) + Text Few-shot + 9 Image Few-shot",
                "temperature": 0,
                "max_tokens": 3000,
                "purpose": "消融实验：测试文本+图像few-shot组合效果",
                "output_directory": self.output_dir,
                "fewshot_config": {
                    "text_fewshot_examples": 3,
                    "image_fewshot_examples": 9,
                    "image_sequences": 3,
                    "images_per_sequence": 3
                },
                "enhanced_characteristics": [
                    "4个详细任务（与run8-rerun相同）",
                    "完整的验证流程",
                    "3个文本few-shot examples",
                    "9张鬼探头图像few-shot examples",
                    "before-during-after序列模式",
                    "严格的分类标准",
                    "Temperature=0确保一致性",
                    "专注ghost probing检测"
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

**Sequence 1 (Images 1-3): Intersection Ghost Probing**
- Image 1 (Before): Normal scene with potential hidden threat behind obstruction
- Image 2 (During): Critical moment when person emerges from behind obstruction
- Image 3 (After): Dangerous situation with visible threat requiring immediate action

**Sequence 2 (Images 4-6): Building Emergence**
- Image 4 (Before): Person hidden behind building structure
- Image 5 (During): Person emerging from building into vehicle path
- Image 6 (After): Full emergence creating collision risk

**Sequence 3 (Images 7-9): Blind Spot Emergence**
- Image 7 (Before): Normal driving scene with hidden potential threat
- Image 8 (During): Critical emergence moment from blind spot
- Image 9 (After): Threat fully visible requiring emergency response

These visual examples demonstrate the classic "ghost probing" pattern:
- **BEFORE**: Normal scene with hidden potential threat
- **DURING**: Critical moment when object emerges from obstruction
- **AFTER**: Dangerous situation with visible threat requiring immediate action

Use both the text examples and visual patterns to guide your analysis.

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

Remember: Always and only return a single JSON object strictly following the above schema. Be incredibly detailed in your analysis, especially for ghost probing detection. Use both the text examples and visual patterns shown in the reference images as guidance.'''
    
    def extract_frames_from_video(self, video_path, video_id, frame_interval=10, frames_per_interval=10):
        """从视频提取帧 - 与run8-rerun完全相同的逻辑"""
        frames = []
        frame_paths = []
        
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            # 创建临时目录存储帧
            frames_dir = os.path.join(self.output_dir, "frames_temp")
            os.makedirs(frames_dir, exist_ok=True)
            
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
        
        self.logger.info(f"🎬 开始分析视频: {video_id}")
        
        # 提取帧
        frames, frame_paths, duration = self.extract_frames_from_video(video_path, video_id)
        
        if not frame_paths:
            self.logger.error(f"❌ 无法提取帧: {video_id}")
            self.results["failed_analyses"] += 1
            return None
        
        # 生成prompt（包含文本few-shot + 9张图像的描述）
        prompt = self.get_run8_rerun_plus_image_prompt(video_id)
        
        self.logger.info(f"📤 发送API请求: {video_id} (with 9 few-shot images)")
        
        # 发送请求（few-shot图像 + 视频帧）
        response = self.send_azure_openai_request(prompt, frame_paths)
        
        if response:
            try:
                # 清理响应，移除markdown代码块标记
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                # 尝试解析JSON
                json_response = json.loads(cleaned_response)
                
                # 保存结果
                result_file = os.path.join(self.output_dir, f"actionSummary_{video_id}.json")
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
                raw_file = os.path.join(self.output_dir, f"raw_response_{video_id}.txt")
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

    def process_video_batch(self, video_list, max_videos=None):
        """批量处理视频"""
        if max_videos:
            video_list = video_list[:max_videos]
            
        self.logger.info(f"🚀 开始批量处理 {len(video_list)} 个视频")
        
        for i, video_path in enumerate(tqdm.tqdm(video_list, desc="处理视频")):
            self.logger.info(f"📊 进度: {i+1}/{len(video_list)}")
            
            result = self.process_single_video(video_path)
            
            # 短暂延迟避免API限制
            if i < len(video_list) - 1:
                time.sleep(3)
                
        # 保存最终结果
        summary_file = os.path.join(self.output_dir, f"experiment_summary_{self.timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"📊 实验完成")
        self.logger.info(f"成功: {self.results['successful_analyses']}")
        self.logger.info(f"失败: {self.results['failed_analyses']}")
        self.logger.info(f"总计: {len(video_list)}")
        self.logger.info(f"📁 结果保存至: {summary_file}")
        
        return self.results

    def retry_failed_videos(self):
        """重试失败的视频"""
        failed_videos = []
        for error in self.results.get("processing_errors", []):
            video_id = error["video_id"]
            # 重新构建视频路径
            video_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/{video_id}.avi"
            if os.path.exists(video_path):
                failed_videos.append(video_path)
                self.logger.info(f"🔄 准备重试失败的视频: {video_id}")
        
        if failed_videos:
            self.logger.info(f"🔄 开始重试 {len(failed_videos)} 个失败的视频")
            # 清除之前的错误记录
            original_failed_count = self.results["failed_analyses"]
            self.results["processing_errors"] = []
            self.results["failed_analyses"] = 0
            
            # 重试失败的视频
            for video_path in failed_videos:
                result = self.process_single_video(video_path)
                time.sleep(3)  # API限制延迟
            
            # 更新最终汇总
            summary_file = os.path.join(self.output_dir, f"experiment_summary_{self.timestamp}.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"🔄 重试完成!")
            self.logger.info(f"重试后成功: {self.results['successful_analyses']}")
            self.logger.info(f"重试后失败: {self.results['failed_analyses']}")
        else:
            self.logger.info("✅ 没有需要重试的失败视频")

if __name__ == "__main__":
    print("🚀 Run8-Rerun + Image Few-shot Experiment")
    print("基于run8-rerun的文本few-shot + 9张鬼探头图像few-shot")
    
    experiment = Run8RerunPlusImageFewshot()
    print(f"📁 输出目录: {experiment.output_dir}")
    print(f"🎯 Few-shot图像: {len(experiment.fewshot_images)}张")
    print("实验配置完成，可以开始处理视频")
    
    # 处理DADA-100的100个视频
    video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos"
    video_files = glob.glob(os.path.join(video_dir, "*.avi"))
    
    # 处理100个视频
    sorted_videos = sorted(video_files)[:100]  # 确保只处理前100个
    print(f"🎯 开始处理DADA-100的{len(sorted_videos)}个视频")
    
    results = experiment.process_video_batch(sorted_videos)
    
    # 重试失败的视频
    if results["failed_analyses"] > 0:
        print(f"🔄 发现 {results['failed_analyses']} 个失败的视频，开始重试...")
        experiment.retry_failed_videos()
    
    print("实验完成！")