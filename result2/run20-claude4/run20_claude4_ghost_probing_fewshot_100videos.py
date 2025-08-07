#!/usr/bin/env python3
"""
Run 20: Claude 4 Ghost Probing Detection with Few-shot Learning (100 Videos)
基于Run 19，使用真正的Claude 4模型 (claude-sonnet-4-20250514)
配置：Paper_Batch架构 + Few-shot Examples + Temperature=0
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
import http.client
import traceback
import sys

# 加载环境变量
load_dotenv()

class Claude4Run20GhostProbingFewshot:
    def __init__(self, output_dir, chunk_size=10):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logging()
        self.setup_claude_api()
        self.load_ground_truth()
        self.initialize_results()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        log_filename = os.path.join(self.output_dir, f"run20_claude4_ghost_probing_fewshot_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Run 20: Claude 4 Ghost Probing Detection with Few-shot Learning (100 Videos) 开始")
        
    def setup_claude_api(self):
        """设置Claude 4 API"""
        # 使用相同的API配置，但使用真正的Claude 4模型
        self.api_host = "globalai.vip"
        self.api_key = os.environ.get("CLAUDE_API_KEY", "")
        if not self.api_key:
            raise ValueError("CLAUDE_API_KEY未设置，请在.env文件中添加")
        
        self.logger.info(f"Claude 4 API配置成功")
        self.logger.info(f"Host: {self.api_host}")
        self.logger.info(f"Model: claude-sonnet-4-20250514 (真正的Claude 4)")
        self.logger.info(f"Temperature: 0, Enhanced with Few-shot Examples")
        
    def load_ground_truth(self):
        """加载ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/labels.csv"
        self.ground_truth = pd.read_csv(gt_path)
        self.logger.info(f"加载ground truth标签: {len(self.ground_truth)}个视频")
        
    def initialize_results(self):
        """初始化结果结构"""
        self.results = {
            "experiment_info": {
                "run_id": "Run 20",
                "timestamp": self.timestamp,
                "video_count": 100,
                "model": "Claude 4 (claude-sonnet-4-20250514)",
                "prompt_version": "Paper_Batch Complex (4-Task) + Few-shot Examples",
                "temperature": 0,
                "max_tokens": 3000,
                "purpose": "真正的Claude 4模型测试，使用与Run 8完全相同的prompt和few-shot配置",
                "output_directory": self.output_dir,
                "prompt_characteristics": [
                    "4个详细任务",
                    "复杂验证流程",
                    "详细的few-shot examples",
                    "严格的分类标准",
                    "Temperature=0确保一致性",
                    "专注ghost probing检测",
                    "100视频完整评估",
                    "与Run 8完全相同的prompt",
                    "真正的Claude 4模型验证"
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
    
    def get_paper_batch_prompt_with_fewshot(self, video_id, frame_interval=10, frames_per_interval=10):
        """获取包含Few-shot Examples的Paper_Batch prompt - 与Run 8/19完全相同"""
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

**Few-shot Examples:**

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

Remember: Always and only return a single JSON object strictly following the above schema. Be incredibly detailed in your analysis, especially for ghost probing detection. Use the examples above as guidance for the level of detail and accuracy expected.'''
    
    def send_claude4_request(self, prompt, images):
        """发送Claude 4 API请求 - 使用真正的Claude 4模型"""
        try:
            # 准备消息内容 - 使用OpenAI兼容格式
            content = [{"type": "text", "text": prompt}]
            
            # 添加图像 - 使用OpenAI格式
            for image_path in images:
                try:
                    with open(image_path, 'rb') as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_string}"
                            }
                        })
                except Exception as e:
                    self.logger.error(f"图像编码失败 {image_path}: {str(e)}")
                    continue
            
            # 构建请求数据 - 使用真正的Claude 4模型ID
            payload = json.dumps({
                "model": "claude-sonnet-4-20250514",
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 3000,
                "temperature": 0
            })
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'Host': 'globalai.vip',
                'Connection': 'keep-alive'
            }
            
            # 发送请求
            conn = http.client.HTTPSConnection(self.api_host)
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read()
            
            response_data = json.loads(data.decode("utf-8"))
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                self.logger.error(f"API响应格式错误: {response_data}")
                return None
                
        except Exception as e:
            self.logger.error(f"Claude 4 API调用失败: {str(e)}")
            return None
    
    def analyze_with_claude4(self, video_path, video_id):
        """使用Claude 4分析视频"""
        try:
            # 提取帧
            frames = self.extract_frames_from_video(video_path)
            if not frames:
                return None
            
            # 生成prompt
            prompt = self.get_paper_batch_prompt_with_fewshot(video_id)
            
            # 发送API请求
            result = self.send_claude4_request(prompt, frames)
            
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
        """运行实验"""
        # 从ground truth文件中获取完整的100个视频列表
        test_videos = self.ground_truth['video_id'].tolist()
        
        self.logger.info(f"开始Run 20实验，处理 {len(test_videos)} 个视频")
        
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
                
                # 分析视频
                result = self.analyze_with_claude4(video_path, video_id)
                
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
                continue
        
        # 保存最终结果
        self.save_final_results()
        self.generate_performance_metrics()
        
    def save_intermediate_results(self, processed_count):
        """保存中间结果"""
        intermediate_file = os.path.join(self.output_dir, f"run20_intermediate_{processed_count}videos_{self.timestamp}.json")
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"中间结果已保存: {intermediate_file}")
    
    def save_final_results(self):
        """保存最终结果"""
        final_file = os.path.join(self.output_dir, f"run20_final_results_{self.timestamp}.json")
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
            "total_videos": len(self.results["detailed_results"])
        }
        
        self.logger.info("=== Run 20 Performance Metrics ===")
        self.logger.info(f"精确度: {precision:.3f} ({precision*100:.1f}%)")
        self.logger.info(f"召回率: {recall:.3f} ({recall*100:.1f}%)")
        self.logger.info(f"F1分数: {f1:.3f} ({f1*100:.1f}%)")
        self.logger.info(f"准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
        self.logger.info(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, ERROR: {errors}")
        
        # 保存指标
        metrics_file = os.path.join(self.output_dir, f"run20_metrics_{self.timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 创建输出目录
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run20-claude4"
    
    # 运行实验
    experiment = Claude4Run20GhostProbingFewshot(output_dir)
    experiment.run_experiment()
    
    print("🎯 Run 20: Claude 4 (claude-sonnet-4-20250514) Ghost Probing Detection (100 Videos) 实验完成!")
    print(f"📁 结果保存在: {output_dir}")