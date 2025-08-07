#!/usr/bin/env python3
"""
快速完成Run 7的剩余视频处理
采用5个视频小批次处理，确保稳定性
"""

import sys
import os
import json
import time
import datetime
import pandas as pd
from dotenv import load_dotenv
import cv2
from moviepy.editor import VideoFileClip
import base64
import requests
import logging
import re
import traceback

# 加载环境变量
load_dotenv()

class FastRun7Completer:
    def __init__(self):
        self.output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run7-gpt4o-paper-batch-temp0"
        self.setup_logging()
        self.setup_api()
        self.load_ground_truth()
        self.load_existing_results()
        
    def setup_logging(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f"run7_fast_completion_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.timestamp = timestamp
        
    def setup_api(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.endpoint = os.environ.get("VISION_ENDPOINT")
        self.deployment = os.environ.get("VISION_DEPLOYMENT_NAME", "gpt-4o-global")
        
    def load_ground_truth(self):
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
        self.ground_truth = pd.read_csv(gt_path, sep='\t')
        
    def load_existing_results(self):
        # 找到最新的中间结果文件
        files = [f for f in os.listdir(self.output_dir) if f.startswith("run7_intermediate_")]
        if not files:
            self.results = {"experiment_info": {"run_id": "Run 7", "timestamp": self.timestamp}, "detailed_results": []}
            return
            
        latest_file = sorted(files)[-1]
        with open(os.path.join(self.output_dir, latest_file), 'r', encoding='utf-8') as f:
            self.results = json.load(f)
            
        processed_videos = {r["video_id"] for r in self.results["detailed_results"]}
        self.logger.info(f"已处理视频: {len(processed_videos)}个")
        
    def get_processed_video_ids(self):
        return {r["video_id"] for r in self.results["detailed_results"]}
        
    def extract_frames(self, video_path):
        frames_dir = os.path.join(self.output_dir, "frames_temp")
        os.makedirs(frames_dir, exist_ok=True)
        
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            frames = []
            frame_interval = 10
            frames_per_interval = 10
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
    
    def get_prompt(self, video_id):
        return f'''You are VideoAnalyzerGPT analyzing a series of SEQUENCIAL images taken from a video, where each image represents a consecutive moment in time.Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of 10 seconds of audio from a video,
as well as as 10 frames split evenly throughout 10 seconds.
You are to generate and provide a Current Action Summary of the video you are considering (10
frames over 10 seconds), which is generated from your analysis of each frame (10 in total),
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
    "End_Timestamp": "10.0s",
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
    
    def send_request(self, prompt, images):
        encoded_images = []
        for image_path in images:
            try:
                with open(image_path, 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    encoded_images.append(encoded_string)
            except Exception as e:
                continue
        
        if not encoded_images:
            return None
            
        content = [{"type": "text", "text": prompt}]
        for encoded_image in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
        
        data = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 3000,
            "temperature": 0
        }
        
        headers = {
            "Content-Type": "application/json", 
            "api-key": self.api_key
        }
        
        try:
            response = requests.post(
                f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version=2024-02-01",
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
    
    def analyze_video(self, video_path, video_id):
        try:
            frames = self.extract_frames(video_path)
            if not frames:
                return None
                
            prompt = self.get_prompt(video_id)
            result = self.send_request(prompt, frames)
            
            # 清理临时文件
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    
            return result
        except Exception as e:
            self.logger.error(f"视频分析失败 {video_id}: {str(e)}")
            return None
    
    def extract_key_actions(self, result_text):
        try:
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            result_json = json.loads(result_text)
            return result_json.get('key_actions', '').lower()
        except:
            key_actions_match = re.search(r'"key_actions":\s*"([^"]*)"', result_text)
            if key_actions_match:
                return key_actions_match.group(1).lower()
            return result_text.lower()
    
    def evaluate_result(self, video_id, key_actions, ground_truth_label):
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
    
    def complete_experiment(self):
        processed_videos = self.get_processed_video_ids()
        
        # 获取需要处理的视频
        remaining_videos = []
        for _, row in self.ground_truth.head(100).iterrows():
            if row['video_id'] not in processed_videos:
                remaining_videos.append(row)
        
        self.logger.info(f"需要处理剩余视频: {len(remaining_videos)}个")
        
        if not remaining_videos:
            self.logger.info("所有视频已处理完成!")
            self.generate_final_report()
            return
        
        # 分批处理剩余视频
        batch_size = 5
        successful = 0
        failed = 0
        
        for i in range(0, len(remaining_videos), batch_size):
            batch = remaining_videos[i:i+batch_size]
            self.logger.info(f"处理批次 {i//batch_size + 1}: 视频 {i+1}-{min(i+batch_size, len(remaining_videos))}")
            
            for row in batch:
                video_id = row['video_id']
                ground_truth_label = row['ground_truth_label']
                video_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/{video_id}"
                
                if not os.path.exists(video_path):
                    self.logger.warning(f"视频文件不存在: {video_path}")
                    failed += 1
                    continue
                
                current_total = len(self.results["detailed_results"]) + 1
                self.logger.info(f"处理视频 {current_total}/99: {video_id}")
                
                result = self.analyze_video(video_path, video_id)
                if result:
                    key_actions = self.extract_key_actions(result)
                    evaluation = self.evaluate_result(video_id, key_actions, ground_truth_label)
                    successful += 1
                else:
                    key_actions = ""
                    evaluation = "ERROR"
                    failed += 1
                
                video_result = {
                    "video_id": video_id,
                    "ground_truth": ground_truth_label,
                    "result": result,
                    "key_actions": key_actions,
                    "evaluation": evaluation
                }
                
                self.results["detailed_results"].append(video_result)
                time.sleep(1)  # 简短延迟
            
            # 保存中间结果
            self.save_intermediate_results()
            self.logger.info(f"批次完成，总进度: {len(self.results['detailed_results'])}/99")
        
        self.logger.info(f"实验完成! 成功: {successful}, 失败: {failed}")
        self.generate_final_report()
    
    def save_intermediate_results(self):
        result_file = os.path.join(self.output_dir, f"run7_intermediate_{self.timestamp}.json")
        self.results["experiment_info"]["timestamp"] = self.timestamp
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
    
    def generate_final_report(self):
        # 计算最终性能指标
        evals = [r['evaluation'] for r in self.results['detailed_results']]
        from collections import Counter
        eval_counts = Counter(evals)
        
        tp = eval_counts.get('TP', 0)
        fp = eval_counts.get('FP', 0)
        tn = eval_counts.get('TN', 0)
        fn = eval_counts.get('FN', 0)
        errors = eval_counts.get('ERROR', 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        total_videos = len(self.results["detailed_results"])
        successful_videos = total_videos - errors
        
        # 保存最终结果
        final_file = os.path.join(self.output_dir, f"run7_final_results_{self.timestamp}.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 打印最终统计
        print(f"\n{'='*60}")
        print("Run 7: GPT-4o + Paper_Batch (Temperature=0) 实验完成!")
        print(f"{'='*60}")
        print(f"处理视频总数: {total_videos}/99")
        print(f"成功处理: {successful_videos}/{total_videos} ({successful_videos/total_videos*100:.1f}%)")
        print(f"精确度: {precision:.3f} ({precision*100:.1f}%)")
        print(f"召回率: {recall:.3f} ({recall*100:.1f}%)")
        print(f"F1分数: {f1:.3f} ({f1*100:.1f}%)")
        print(f"准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, ERROR: {errors}")
        
        # 与Run 6对比
        run6_f1 = 0.636
        run6_precision = 0.554
        run6_recall = 0.745
        
        print(f"\n与Run 6对比 (Temperature=0.3):")
        print(f"F1分数: {f1:.3f} vs {run6_f1:.3f} ({f1-run6_f1:+.3f})")
        print(f"精确度: {precision:.3f} vs {run6_precision:.3f} ({precision-run6_precision:+.3f})")
        print(f"召回率: {recall:.3f} vs {run6_recall:.3f} ({recall-run6_recall:+.3f})")
        
        return {
            'total_videos': total_videos,
            'successful_videos': successful_videos,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'errors': errors
        }

if __name__ == "__main__":
    completer = FastRun7Completer()
    print("开始快速完成Run 7实验...")
    print("="*60)
    
    try:
        completer.complete_experiment()
    except KeyboardInterrupt:
        print("\n实验被中断，已保存当前进度")
    except Exception as e:
        print(f"实验出现错误: {e}")
        traceback.print_exc()