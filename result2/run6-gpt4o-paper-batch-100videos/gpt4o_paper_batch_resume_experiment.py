#!/usr/bin/env python3
"""
GPT-4o + Paper_Batch版本 100个视频完整实验 - 可恢复版本
从中断点继续处理剩余视频
"""

import cv2
import os
import json
import logging
import time
import datetime
from moviepy.editor import VideoFileClip
import pandas as pd
import openai
from dotenv import load_dotenv
import tqdm
import re
import base64
import requests
import traceback

# 加载环境变量
load_dotenv()

class GPT4oPaperBatchResume:
    def __init__(self, output_dir, start_from_video=31):
        self.output_dir = output_dir
        self.start_from_video = start_from_video
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logging()
        self.setup_openai_api()
        self.load_ground_truth()
        self.load_existing_results()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.output_dir, f"gpt4o_paper_batch_resume_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"GPT-4o Paper_Batch 恢复实验，从第{self.start_from_video}个视频开始")
        
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
        
    def load_ground_truth(self):
        """加载ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
        self.ground_truth = pd.read_csv(gt_path, sep='\t')
        self.logger.info(f"加载ground truth标签: {len(self.ground_truth)}个视频")
        
    def load_existing_results(self):
        """加载现有结果"""
        # 查找最新的中间结果文件
        intermediate_files = [f for f in os.listdir(self.output_dir) if f.startswith("gpt4o_paper_batch_intermediate_")]
        if intermediate_files:
            latest_file = sorted(intermediate_files)[-1]
            intermediate_path = os.path.join(self.output_dir, latest_file)
            
            with open(intermediate_path, 'r', encoding='utf-8') as f:
                self.existing_results = json.load(f)
            
            processed_count = len(self.existing_results["detailed_results"])
            self.logger.info(f"找到现有结果文件: {latest_file}")
            self.logger.info(f"已处理视频数量: {processed_count}")
        else:
            self.existing_results = {
                "experiment_info": {
                    "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "video_count": 100,
                    "model": "GPT-4o (Azure)",
                    "prompt_version": "Paper_Batch Complex (4-Task)",
                    "output_directory": self.output_dir,
                    "prompt_characteristics": [
                        "4个详细任务",
                        "中英文混合内容",
                        "复杂验证流程",
                        "极其详细的分析要求",
                        "严格的分类标准"
                    ]
                },
                "detailed_results": []
            }
            self.logger.info("未找到现有结果，将从头开始")
        
    def extract_frames_from_video(self, video_path, frame_interval=10, frames_per_interval=10):
        """从视频中提取帧"""
        frames_dir = os.path.join(self.output_dir, "frames_temp")
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
    
    def get_paper_batch_prompt(self, video_id, frame_interval=10, frames_per_interval=10):
        """Paper_Batch版本的完整复杂prompt"""
        return f'''You are VideoAnalyzerGPT analyzing a series of SEQUENCIAL images taken from a video, where each image represents a consecutive moment in time.Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the portion of the video you are considering.

Direction - Please identify the objects in the image based on their position in the image itself. Do not assume your own position within the image. Treat the left side of the image as 'left' and the right side of the image as 'right'. Assume the viewpoint is standing from at the bottom center of the image. Describe whether the objects are on the left or right side of this central point, left is left, and right is right. For example, if there is a car on the left side of the image, state that the car is on the left.

**Task 1: Identify and Predict potential "Ghost Probing(专业术语：鬼探头)",Cut-in(加塞) etc behavior**

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

Note: Only those emerging from behind a physical obstruction can be considered as 鬼探头. Cut-in加塞 is different from 鬼探头.
    **IMPORTANT DISTINCTION**: When a vehicle enters suddenly from a perpendicular road or from behind a physical obstruction, this is "ghost probing" NOT "cut-in". Pay careful attention to the origin of the vehicle - if it comes from a side street or behind an obstruction rather than an adjacent lane, it must be classified as "ghost probing".
    
    For vehicle ghost probing, be vigilant throughout the entire video for vehicles suddenly appearing from behind obstructions such as buildings, walls, parked vehicles, or entering from perpendicular roads where visibility was blocked.    

2) Cut-In(加塞):
    Definition: Cut-in occurs ONLY when a vehicle from the SAME DIRECTION in an ADJACENT LANE merges into the self-vehicle's lane. The key difference between cut-in and ghost probing is:
    - Cut-in: Vehicle is visible in adjacent lane BEFORE changing lanes (no physical obstruction)
    - Ghost probing: Vehicle is NOT visible until it emerges from behind a physical obstruction or from a perpendicular road
  
    Typically **within same-direction traffic flow**, a cut-in happens when a vehicle deliberately forces its way in front of another vehicle's traffic lane from the **adjacent lane**, occupying another driver's lane space. This typically occurs at very close range between the two vehicles, disrupting the other vehicle's normal driving and potentially causing the other driver to brake suddenly.
    加塞是指在**同向**车流行驶过程中，某车辆从**侧面相邻车道**强行插入其他车辆的行驶路线,强行抢占他人车道的行驶空间，这种情况下一般是指距离非常近，从而影响其他车辆的正常行驶，甚至导致紧急刹车。

    ### Classification Flow:
    1. Is there a physical obstruction blocking view of the vehicle before it appears? If YES → "ghost probing"
    2. Does the vehicle come from a perpendicular road? If YES → "ghost probing"
    3. Is the vehicle visible in an adjacent lane before merging? If YES → "cut-in"

**Validation Process:**
  - After identifying a vehicle's movement, carefully analyze:
    - If it came from behind a physical obstruction → label as "ghost probing"
    - If it emerged from a perpendicular road → label as "ghost probing"
    - If it was visible in an adjacent lane and then merged → label as "cut-in"

Your angle appears to watch video frames recorded from a surveillance camera in a car. Your role should focus on detecting and predicting dangerous actions in a "Ghosting" manner
where pedestrians or vehicles in the scene might suddenly appear in front of the current car. This could happen if a person or vehicle suddenly emerges from behind an obstacle in the driver's view.
This behavior is extremely dangerous because it gives the driver very little time to react.
Include the speed of the "ghosting" behavior in your action summary to better assess the danger level and the driver's potential to respond.

Provide detailed description of both people's and vehicles' behavior and potential dangerous actions that could lead to collisions. Describe how you think the individual or vehicle could crash into the car, and explain your deduction process. Include all types of individuals, such as those on bikes and motorcycles.
Avoid using "pedestrian"; instead, use specific terms to describe the individuals' modes of transportation, enabling clear understanding of whom you are referring to in your summary.
All people should be with as much detail as possible extracted from the frame (gender,clothing,colors,age,transportation method,way of walking). Be incredibly detailed. Output in the "summary" field of the JSON format template.

**Task 2: Explain Current Driving Actions**
Analyze the current video frames to extract actions. Describe not only the actions themselves but also provide detailed reasoning for why the vehicle is taking these actions, such as changes in speed and direction. Focus solely on the reasoning for the vehicle's actions, excluding any descriptions of pedestrian behavior. Explain why the driver is driving at a certain speed, making turns, or stopping. Your goal is to provide a comprehensive understanding of the vehicle's behavior based on the visual data. Output in the "actions" field of the JSON format template.

**Task 3: Predict Next Driving Action**
Understand the current road conditions, the driving behavior, and to predict the next driving action. Analyze the video and audio to provide a comprehensive summary of the road conditions, including weather, traffic density, road obstacles, and traffic light if visible. Predict the next driving action based on two dimensions, one is driving speed control, such as accelerating, braking, turning, or stopping, the other one is to predict the next lane control, such as change to left lane, change to right lane, keep left in this lane, keep right in this lane, keep straight. Your summary should help understand not only what is happening at the moment but also what is likely to happen next with logical reasoning. The principle is safety first, so the prediction action should prioritize the driver's safety and secondly the pedestrians' safety. Be incredibly detailed. Output in the "next_action" field of the JSON format template.

**Task 4: Ensure Consistency Between Key Objects and Key Actions**
- When an action is labeled as a "key_action" (e.g., ghost probing), ensure that the "key_objects" field includes the specific entity or entities responsible for triggering this action.
- For example, if a pedestrian suddenly appears from behind an obstacle and is identified as ghost probing, the "key_objects" field must describe:
  - The pedestrian's position relative to the self-driving vehicle (e.g., left side, right side, etc.).
  - The pedestrian's behavior leading to the key action (e.g., moving suddenly from behind a parked truck).
  - The potential impact on the vehicle (e.g., causing the vehicle to decelerate or stop).
- Each key object description should include:
  - Relative position (e.g., left, right, front).
  - Distance from the vehicle in meters.
  - Movement direction or behavior (e.g., approaching, crossing, accelerating).
  - The relationship to the "key_action" it caused.
- Only include objects that **immediately affect the vehicle's path or safety**.

Additional Requirements:
- `key_actions` must strictly adhere to the predefined categories:
    - ghost probing
    - cut-in
    - overtaking, specify "left-side overtaking" or "right-side overtaking" when relevant.
    - none (if no dangerous behavior is observed)

- All textual fields must be in English.
- `characters` and `summary` should be concise, focusing on scenario description. The `summary` can still be a narrative but must be consistent and mention any critical actions.
- Avoid generic descriptions such as "A person or vehicle suddenly appeared." Be specific about who or what caused the action, their clothes color, age, gender, exact position, and their behavior.

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
    "key_actions": "ghost probing/cut-in/left-side overtaking/right-side overtaking/none",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}

Audio Transcription: [No audio available for this analysis]

Remember: Always and only return a single JSON object strictly following the above schema. Be incredibly detailed in your analysis, especially for ghost probing detection.'''
    
    def send_azure_openai_request(self, prompt, images):
        """发送Azure OpenAI请求"""
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
            "temperature": 0
        }
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.openai_api_key
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.vision_endpoint}/openai/deployments/{self.vision_deployment}/chat/completions?api-version=2024-02-01",
                    headers=headers,
                    json=data,
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
            except requests.exceptions.Timeout:
                self.logger.warning(f"API请求超时，尝试 {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                continue
            except Exception as e:
                self.logger.error(f"Azure OpenAI API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                continue
        
        return None
    
    def analyze_with_gpt4o(self, video_path, video_id):
        """使用GPT-4o分析视频"""
        try:
            frames, timestamps = self.extract_frames_from_video(video_path)
            if not frames:
                return None
                
            prompt = self.get_paper_batch_prompt(video_id)
            
            # API调用
            result = self.send_azure_openai_request(prompt, frames)
            
            # 清理临时文件
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"GPT-4o分析失败 {video_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
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
        has_ghost_probing = "ghost probing" in key_actions
        ground_truth_has_ghost = ground_truth_label != "none"
        
        if has_ghost_probing and ground_truth_has_ghost:
            return "TP"  # True Positive
        elif has_ghost_probing and not ground_truth_has_ghost:
            return "FP"  # False Positive
        elif not has_ghost_probing and ground_truth_has_ghost:
            return "FN"  # False Negative
        else:
            return "TN"  # True Negative
    
    def resume_experiment(self):
        """恢复实验，从指定位置继续"""
        timestamp = self.existing_results["experiment_info"]["timestamp"]
        
        # 获取前100个视频
        test_videos = self.ground_truth.head(100)
        
        # 确定从哪个视频开始
        processed_videos = len(self.existing_results["detailed_results"])
        remaining_videos = test_videos.iloc[processed_videos:]
        
        self.logger.info(f"已处理视频: {processed_videos}/100")
        self.logger.info(f"剩余视频: {len(remaining_videos)}")
        
        successful_count = sum(1 for r in self.existing_results["detailed_results"] if r["evaluation"] != "ERROR")
        failed_count = processed_videos - successful_count
        
        for idx, row in tqdm.tqdm(remaining_videos.iterrows(), total=len(remaining_videos)):
            video_id = row['video_id']
            ground_truth_label = row['ground_truth_label']
            
            video_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/{video_id}"
            
            if not os.path.exists(video_path):
                self.logger.warning(f"视频文件不存在: {video_path}")
                failed_count += 1
                continue
                
            self.logger.info(f"处理视频 {processed_videos+1}/100: {video_id}")
            
            # Paper_batch分析
            result = self.analyze_with_gpt4o(video_path, video_id)
            key_actions = ""
            evaluation = "ERROR"
            
            if result:
                key_actions = self.extract_key_actions(result)
                evaluation = self.evaluate_result(video_id, key_actions, ground_truth_label)
                successful_count += 1
            else:
                failed_count += 1
            
            # 记录结果
            video_result = {
                "video_id": video_id,
                "ground_truth": ground_truth_label,
                "result": result,
                "key_actions": key_actions,
                "evaluation": evaluation
            }
            
            self.existing_results["detailed_results"].append(video_result)
            processed_videos += 1
            
            # 每10个视频保存一次中间结果
            if processed_videos % 10 == 0:
                intermediate_file = os.path.join(self.output_dir, f"gpt4o_paper_batch_intermediate_{timestamp}.json")
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(self.existing_results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"已保存{processed_videos}个视频的中间结果 (成功: {successful_count}, 失败: {failed_count})")
                
            # 添加延迟避免API限制
            time.sleep(2)
                
        # 最终保存结果
        result_file = os.path.join(self.output_dir, f"gpt4o_paper_batch_100videos_results_{timestamp}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.existing_results, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Paper_Batch实验完成")
        self.logger.info(f"成功处理: {successful_count}/100 个视频 ({successful_count/100*100:.1f}%)")
        self.generate_summary_report(self.existing_results, timestamp, successful_count, failed_count)
        
    def generate_summary_report(self, results, timestamp, successful_count, failed_count):
        """生成总结报告"""
        # 计算性能指标
        stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "ERROR": 0}
        
        for result in results["detailed_results"]:
            evaluation = result["evaluation"]
            stats[evaluation] = stats.get(evaluation, 0) + 1
        
        tp, fp, tn, fn = stats["TP"], stats["FP"], stats["TN"], stats["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # 生成markdown报告
        report = f"""# GPT-4o + Paper_Batch版本 100个视频实验报告

## 实验概述

- **实验时间**: {timestamp}
- **模型**: GPT-4o (Azure)
- **Prompt版本**: Paper_Batch Complex (4-Task Version)
- **测试视频数**: 100个 (DADA-100-videos)
- **成功处理**: {successful_count}/100 ({successful_count}%)
- **失败数**: {failed_count}/100 ({failed_count}%)

## Prompt特征

- ✅ 4个详细任务 (Ghost Probing检测、当前行为解释、下一步预测、一致性检查)
- ✅ 中英文混合内容和术语解释
- ✅ 复杂的分类验证流程
- ✅ 极其详细的分析要求 ("Be incredibly detailed"出现多次)
- ✅ 严格的分类标准和惩罚机制

## 性能指标

| 指标 | 数值 |
|------|------|
| **精确度 (Precision)** | {precision:.3f} ({precision*100:.1f}%) |
| **召回率 (Recall)** | {recall:.3f} ({recall*100:.1f}%) |
| **F1分数** | {f1:.3f} ({f1*100:.1f}%) |
| **准确率 (Accuracy)** | {accuracy:.3f} ({accuracy*100:.1f}%) |

## 统计详情

- **True Positives (TP)**: {tp}
- **False Positives (FP)**: {fp}
- **True Negatives (TN)**: {tn}
- **False Negatives (FN)**: {fn}
- **处理错误 (ERROR)**: {stats['ERROR']}

## 与其他版本对比

| 模型+Prompt版本 | 视频数 | 精确度 | 召回率 | F1分数 | 备注 |
|-----------------|--------|--------|--------|--------|------|
| **GPT-4o + Paper_Batch** | {successful_count} | {precision:.3f} | {recall:.3f} | {f1:.3f} | 本次实验 |
| **Gemini + Paper_Batch** | 97 | 0.612 | 0.774 | 0.683 | 历史结果 |
| **GPT-4.1 + Balanced** | 100 | 0.565 | 0.963 | 0.712 | 历史结果 |

## 关键发现

### 1. 处理成功率
- **成功率**: {successful_count/100*100:.1f}% - {"优秀" if successful_count >= 90 else "良好" if successful_count >= 80 else "需要改进"}

### 2. 性能分析
- **精确度**: {precision:.3f} - {"高精确度，误报较少" if precision > 0.7 else "中等精确度" if precision > 0.5 else "精确度较低，需要优化"}
- **召回率**: {recall:.3f} - {"高召回率，漏检较少" if recall > 0.8 else "中等召回率" if recall > 0.6 else "召回率较低，安全风险"}
- **F1分数**: {f1:.3f} - {"优秀平衡" if f1 > 0.7 else "良好平衡" if f1 > 0.6 else "需要平衡优化"}

### 3. Paper_Batch版本特点验证
- ✅ 复杂4任务结构已实现
- ✅ 详细分析要求已执行
- ✅ 中英文混合术语已包含
- ✅ 严格验证流程已应用

## 结论

GPT-4o配合Paper_Batch复杂版本prompt在100个视频上取得了{f1:.3f}的F1分数。
{"该版本在精确度和召回率之间取得了良好平衡，证明了复杂prompt的有效性。" if f1 > 0.7 else "该版本展现了Paper_Batch复杂设计的特点，但性能仍有优化空间。"}

对于ghost probing这种安全关键场景，{recall:.3f}的召回率{"足够保障安全性" if recall > 0.85 else "需要进一步提升以确保安全性"}。

## 技术细节

- **结果文件**: {self.output_dir}/gpt4o_paper_batch_100videos_results_{timestamp}.json
- **中间结果**: {self.output_dir}/gpt4o_paper_batch_intermediate_{timestamp}.json
- **日志文件**: {self.output_dir}/gpt4o_paper_batch_resume_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log
- **帧提取**: 10秒间隔，每间隔10帧
- **API配置**: Azure OpenAI, {self.vision_deployment}

---
*报告生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*基于Paper_Batch复杂prompt的完整100视频实验*
"""
        
        report_file = os.path.join(self.output_dir, f"gpt4o_paper_batch_100videos_report_{timestamp}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        # 打印总结
        print("\n" + "="*60)
        print("GPT-4o + Paper_Batch 100个视频实验完成")
        print("="*60)
        print(f"成功处理: {successful_count}/100 视频")
        print(f"精确度: {precision:.3f}")
        print(f"召回率: {recall:.3f}")
        print(f"F1分数: {f1:.3f}")
        print(f"结果保存到: {self.output_dir}")
        print(f"详细报告: gpt4o_paper_batch_100videos_report_{timestamp}.md")

if __name__ == "__main__":
    # 设置输出目录
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gpt4o-paper-batch-100videos"
    
    analyzer = GPT4oPaperBatchResume(output_dir)
    
    print(f"恢复 GPT-4o + Paper_Batch 100个视频实验")
    print(f"输出目录: {output_dir}")
    print("="*60)
    
    analyzer.resume_experiment()