#!/usr/bin/env python3
"""
GPT-4o Prompt对比实验
对比Early版本 vs GPT4o-Balanced版本在前20个视频上的表现
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

# 加载环境变量
load_dotenv()

class GPT4oPromptComparison:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logging()
        self.setup_openai_api()
        self.load_ground_truth()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.output_dir, f"gpt4o_comparison_log_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("GPT-4o Prompt对比实验开始")
        
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
        self.logger.info(f"API Key: {self.openai_api_key[:20]}...")
        
    def load_ground_truth(self):
        """加载ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
        self.ground_truth = pd.read_csv(gt_path, sep='\\t')
        self.logger.info(f"加载ground truth标签: {len(self.ground_truth)}个视频")
        
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
    
    def get_early_prompt(self, video_id, frame_interval=10, frames_per_interval=10):
        """Early版本的详细prompt"""
        return f'''You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the portion of the video you are considering.

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

2) Cut-In(加塞):
    Definition: Cut-in occurs ONLY when a vehicle from the SAME DIRECTION in an ADJACENT LANE merges into the self-vehicle's lane. The key difference between cut-in and ghost probing is:
    - Cut-in: Vehicle is visible in adjacent lane BEFORE changing lanes (no physical obstruction)
    - Ghost probing: Vehicle is NOT visible until it emerges from behind a physical obstruction or from a perpendicular road
  
    Typically within same-direction traffic flow, a cut-in happens when a vehicle deliberately forces its way in front of another vehicle's traffic lane from the adjacent lane, occupying another driver's lane space. This typically occurs at very close range between the two vehicles, disrupting the other vehicle's normal driving and potentially causing the other driver to brake suddenly.

### Classification Flow:
1. Is there a physical obstruction blocking view of the vehicle before it appears? If YES → "ghost probing"
2. Does the vehicle come from a perpendicular road? If YES → "ghost probing"
3. Is the vehicle visible in an adjacent lane before merging? If YES → "cut-in"

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
    "key_actions": "ghost probing/cut-in/left-side overtaking/right-side overtaking/none",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}'''
    
    def get_balanced_prompt(self, video_id, frame_interval=10, frames_per_interval=10):
        """GPT4o-Balanced版本的prompt"""
        return f'''You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the video.

IMPORTANT: For ghost probing detection, consider THREE categories:

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
}}'''
    
    def send_azure_openai_request(self, prompt, images):
        """发送Azure OpenAI请求"""
        encoded_images = []
        for image_path in images:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append(encoded_string)
        
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
            "max_tokens": 2000,
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
                json=data
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            self.logger.error(f"Azure OpenAI API调用失败: {str(e)}")
            return None
    
    def analyze_with_gpt4o(self, video_path, video_id, prompt_type="balanced"):
        """使用GPT-4o分析视频"""
        try:
            frames, timestamps = self.extract_frames_from_video(video_path)
            if not frames:
                return None
                
            if prompt_type == "early":
                prompt = self.get_early_prompt(video_id)
            else:
                prompt = self.get_balanced_prompt(video_id)
            
            # API调用
            result = self.send_azure_openai_request(prompt, frames)
            
            # 清理临时文件
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"GPT-4o分析失败 {video_id}: {str(e)}")
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
            key_actions_match = re.search(r'"key_actions":\\s*"([^"]*)"', result_text)
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
    
    def run_comparison_experiment(self):
        """运行对比实验"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 选择前20个视频
        test_videos = self.ground_truth.head(20)
        
        results = {
            "experiment_info": {
                "timestamp": timestamp,
                "video_count": len(test_videos),
                "model": "GPT-4o",
                "comparison": "Early vs Balanced Prompt",
                "output_directory": self.output_dir
            },
            "detailed_results": []
        }
        
        self.logger.info(f"开始对比实验，处理 {len(test_videos)} 个视频")
        
        for idx, row in tqdm.tqdm(test_videos.iterrows(), total=len(test_videos)):
            video_id = row['video_id']
            ground_truth_label = row['ground_truth_label']
            
            video_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/{video_id}"
            
            if not os.path.exists(video_path):
                self.logger.warning(f"視頻文件不存在: {video_path}")
                continue
                
            self.logger.info(f"處理視頻 {idx+1}/{len(test_videos)}: {video_id}")
            
            # Early prompt分析
            early_result = self.analyze_with_gpt4o(video_path, video_id, "early")
            early_key_actions = ""
            early_evaluation = "ERROR"
            
            if early_result:
                early_key_actions = self.extract_key_actions(early_result)
                early_evaluation = self.evaluate_result(video_id, early_key_actions, ground_truth_label)
            
            # Balanced prompt分析
            balanced_result = self.analyze_with_gpt4o(video_path, video_id, "balanced")
            balanced_key_actions = ""
            balanced_evaluation = "ERROR"
            
            if balanced_result:
                balanced_key_actions = self.extract_key_actions(balanced_result)
                balanced_evaluation = self.evaluate_result(video_id, balanced_key_actions, ground_truth_label)
            
            # 記錄結果
            video_result = {
                "video_id": video_id,
                "ground_truth": ground_truth_label,
                "early_prompt": {
                    "result": early_result,
                    "key_actions": early_key_actions,
                    "evaluation": early_evaluation
                },
                "balanced_prompt": {
                    "result": balanced_result,
                    "key_actions": balanced_key_actions,
                    "evaluation": balanced_evaluation
                }
            }
            
            results["detailed_results"].append(video_result)
            
            # 每5個視頻保存一次中間結果
            if (idx + 1) % 5 == 0:
                intermediate_file = os.path.join(self.output_dir, f"gpt4o_comparison_intermediate_{timestamp}.json")
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"已保存{idx+1}個視頻的中間結果")
                
        # 最終保存結果
        result_file = os.path.join(self.output_dir, f"gpt4o_comparison_results_{timestamp}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"對比實驗完成")
        self.generate_comparison_report(results, timestamp)
        
    def generate_comparison_report(self, results, timestamp):
        """生成對比報告"""
        # 計算兩個prompt的性能指標
        early_stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "ERROR": 0}
        balanced_stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "ERROR": 0}
        
        for result in results["detailed_results"]:
            early_stats[result["early_prompt"]["evaluation"]] += 1
            balanced_stats[result["balanced_prompt"]["evaluation"]] += 1
        
        def calculate_metrics(stats):
            tp, fp, tn, fn = stats["TP"], stats["FP"], stats["TN"], stats["FN"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            return {"precision": precision, "recall": recall, "f1_score": f1, "accuracy": accuracy, "stats": stats}
        
        early_metrics = calculate_metrics(early_stats)
        balanced_metrics = calculate_metrics(balanced_stats)
        
        # 生成markdown報告
        markdown_report = f"""# GPT-4o Prompt對比實驗報告

## 實驗概述

- **實驗時間**: {timestamp}
- **模型**: GPT-4o (Azure)
- **測試視頻數**: {len(results["detailed_results"])}
- **對比維度**: Early版本 vs GPT4o-Balanced版本

## 性能對比

### Early Prompt版本
| 指標 | 數值 |
|------|------|
| 精確度 (Precision) | {early_metrics['precision']:.3f} |
| 召回率 (Recall) | {early_metrics['recall']:.3f} |
| F1分數 | {early_metrics['f1_score']:.3f} |
| 準確率 (Accuracy) | {early_metrics['accuracy']:.3f} |

**統計詳情**: {early_metrics['stats']}

### GPT4o-Balanced版本
| 指標 | 數值 |
|------|------|
| 精確度 (Precision) | {balanced_metrics['precision']:.3f} |
| 召回率 (Recall) | {balanced_metrics['recall']:.3f} |
| F1分數 | {balanced_metrics['f1_score']:.3f} |
| 準確率 (Accuracy) | {balanced_metrics['accuracy']:.3f} |

**統計詳情**: {balanced_metrics['stats']}

## 改進效果

| 指標 | Early版本 | Balanced版本 | 改進幅度 |
|------|-----------|-------------|----------|
| 精確度 | {early_metrics['precision']:.3f} | {balanced_metrics['precision']:.3f} | {(balanced_metrics['precision'] - early_metrics['precision']):.3f} |
| 召回率 | {early_metrics['recall']:.3f} | {balanced_metrics['recall']:.3f} | {(balanced_metrics['recall'] - early_metrics['recall']):.3f} |
| F1分數 | {early_metrics['f1_score']:.3f} | {balanced_metrics['f1_score']:.3f} | {(balanced_metrics['f1_score'] - early_metrics['f1_score']):.3f} |

## 結論

{self.generate_conclusion(early_metrics, balanced_metrics)}

實驗數據保存在: `{self.output_dir}`

---
*報告生成時間: {timestamp}*
"""
        
        markdown_file = os.path.join(self.output_dir, f"gpt4o_comparison_report_{timestamp}.md")
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
            
        # 打印總結
        print("\\n" + "="*60)
        print("GPT-4o Prompt對比實驗總結")
        print("="*60)
        print(f"測試視頻數量: {len(results['detailed_results'])}")
        print(f"實驗時間: {timestamp}")
        print("\\nEarly Prompt性能:")
        print(f"  精確度: {early_metrics['precision']:.3f}")
        print(f"  召回率: {early_metrics['recall']:.3f}")
        print(f"  F1分數: {early_metrics['f1_score']:.3f}")
        print("\\nBalanced Prompt性能:")
        print(f"  精確度: {balanced_metrics['precision']:.3f}")
        print(f"  召回率: {balanced_metrics['recall']:.3f}")
        print(f"  F1分數: {balanced_metrics['f1_score']:.3f}")
        print(f"\\n結果已保存到: {self.output_dir}")
        
    def generate_conclusion(self, early_metrics, balanced_metrics):
        """生成結論"""
        f1_improvement = balanced_metrics['f1_score'] - early_metrics['f1_score']
        precision_improvement = balanced_metrics['precision'] - early_metrics['precision']
        
        if f1_improvement > 0.1:
            return "GPT4o-Balanced版本在F1分數上有顯著改進，證明了簡化和平衡策略的有效性。"
        elif precision_improvement > 0.2:
            return "GPT4o-Balanced版本在精確度上有明顯提升，有效減少了誤報率。"
        else:
            return "兩個prompt版本性能相近，需要進一步分析具體差異。"

if __name__ == "__main__":
    # 設置輸出目錄
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gpt4o-prompt-comparison-run5"
    
    analyzer = GPT4oPromptComparison(output_dir)
    
    print(f"開始 GPT-4o Prompt對比實驗")
    print(f"輸出目錄: {output_dir}")
    print("="*60)
    
    analyzer.run_comparison_experiment()