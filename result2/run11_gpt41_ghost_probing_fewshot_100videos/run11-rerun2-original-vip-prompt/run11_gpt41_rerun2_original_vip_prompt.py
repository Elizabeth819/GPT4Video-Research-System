#!/usr/bin/env python3
"""
Run 11 Rerun2: GPT-4.1 Ghost Probing Detection with Original VIP Prompt (100 Videos)
使用原始 /Users/wanmeng/repository/GPT4Video-cobra-auto/VIP/ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper-batch.py 
中的完整详细prompt，重新运行Run 11实验
目的：对比简化prompt和原始详细prompt在GPT-4.1上的性能差异
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

class GPT41Run11Rerun2OriginalVIP:
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
        log_filename = os.path.join(self.output_dir, f"run11_gpt41_rerun2_original_vip_{timestamp}.log")
        
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
        """设置Azure OpenAI API for GPT-4.1"""
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.vision_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
        self.vision_deployment = os.getenv("GPT_4.1_VISION_DEPLOYMENT_NAME", "gpt-4.1")  # 使用GPT-4.1
        
        if not all([self.openai_api_key, self.vision_endpoint, self.vision_deployment]):
            raise ValueError("Azure OpenAI环境变量未设置完整")
            
        self.logger.info(f"Azure OpenAI API配置成功 - GPT-4.1")
        self.logger.info(f"Endpoint: {self.vision_endpoint}")
        self.logger.info(f"Deployment: {self.vision_deployment}")
        self.logger.info(f"Temperature: 0, 使用原始VIP详细Prompt")
        
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
                "run_id": "Run 11 Rerun2",
                "timestamp": self.timestamp,
                "video_count": 100,
                "model": "GPT-4.1 (Azure)",
                "prompt_version": "Original VIP Detailed Prompt (from ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper-batch.py)",
                "temperature": 0,
                "max_tokens": 2000,
                "purpose": "GPT-4.1重新运行2：使用原始VIP详细prompt测试性能，对比简化prompt效果差异",
                "ground_truth_file": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/labels.csv",
                "output_directory": self.output_dir,
                "prompt_characteristics": [
                    "4个详细任务（Ghost Probing + Cut-in + Current Actions + Next Actions）",
                    "极其详细的分类标准和验证流程",
                    "严格的Cut-in vs Ghost Probing区分逻辑",
                    "详细的key_objects一致性要求",
                    "强制英文输出约束",
                    "完整的penalty机制",
                    "Temperature=0确保一致性",
                    "专注ghost probing检测",
                    "100视频完整评估",
                    "使用校正后labels.csv",
                    "GPT-4.1模型验证",
                    "原始VIP详细prompt版本"
                ]
            },
            "detailed_results": []
        }
    
    def load_existing_results(self):
        """加载现有的中间结果"""
        import glob
        
        # 查找最新的中间结果文件
        intermediate_files = glob.glob(os.path.join(self.output_dir, "run11_gpt41_rerun2_intermediate_*videos_*.json"))
        
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
        """从视频中提取帧"""
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

    def get_original_vip_detailed_prompt(self, video_id, frame_interval=10, frames_per_interval=10):
        """获取原始VIP详细prompt - 从ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper-batch.py复制"""
        segment_id_str = "full_video"
        
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
                        Characteristics:
                        A cut-in is defined only when a vehicle merges into the current lane from an adjacent side lane.
                        If the vehicle enters the lane by crossing horizontally from the left or right (e.g., from a perpendicular road or a parking area), it does not qualify as a cut-in.
                        Cut-in特点: 只有从相邻车道侧面插入进当前车道才算cut-in, 如果是从左右手两边的垂直的路上横插过来不算cut-in.
                        ### Key Rules:
                        1. Cut-in occurs ONLY when a vehicle merges from an adjacent side lane.
                        2. Entry from perpendicular or non-adjacent lanes is NOT "cut-in" but potentially "ghost probing".

                        ### Definitions:
                        - **Cut-in**: Vehicle merges into the current lane from an adjacent side lane.
                        - **Ghost probing**: Vehicle enters the current lane from a perpendicular road or emerges from behind a physical obstruction.

                        ### Classification Examples:
                        - **正例 (Cut-in)**:
                        - A car from the adjacent left lane merges into the self-vehicle's lane abruptly.
                        - **反例 (NOT Cut-in, but Ghost Probing)**:
                        - A car enters from a perpendicular road on the right and suddenly appears from behind a physical obstruction.
                        注意: 任何来自垂直侧路的插入且是从遮挡物后面窜出均是"ghost probing"，而非 cut-in。

                        ### Classification Flow:
                        1. Is there a physical obstruction blocking view of the vehicle before it appears? If YES → "ghost probing"
                        2. Does the vehicle come from a perpendicular road? If YES → "ghost probing"
                        3. Is the vehicle visible in an adjacent lane before merging? If YES → "cut-in"

                        ***Key Note***
                        Vehicles entering from a perpendicular road or from behind physical obstructions should never be labeled as "cut-in". These must be classified as "ghost probing" if they create a dangerous situation with minimal reaction time.

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
                    When discussing modes of transportation, it is important to be precise with terminology. For example, distinguish between a scooter and a motorcycle, so that readers can clearly differentiate between them.
                    Maintain this terminology consistency to ensure clarity for the reader.
                    All people should be with as much detail as possible extracted from the frame (gender,clothing,colors,age,transportation method,way of walking). Be incredibly detailed. Output in the "summary" field of the JSON format template.

                    **Task 2: Explain Current Driving Actions**
                    Analyze the current video frames to extract actions. Describe not only the actions themselves but also provide detailed reasoning for why the vehicle is taking these actions, such as changes in speed and direction. Focus solely on the reasoning for the vehicle's actions, excluding any descriptions of pedestrian behavior. Explain why the driver is driving at a certain speed, making turns, or stopping. Your goal is to provide a comprehensive understanding of the vehicle's behavior based on the visual data. Output in the "actions" field of the JSON format template.

                    **Task 3: Predict Next Driving Action**
                    Understand the current road conditions, the driving behavior, and to predict the next driving action. Analyze the video and audio to provide a comprehensive summary of the road conditions, including weather, traffic density, road obstacles, and traffic light if visible. Predict the next driving action based on two dimensions, one is driving speed control, such as accelerating, braking, turning, or stopping, the other one is to predict the next lane control, such as change to left lane, change to right lane, keep left in this lane, keep right in this lane, keep straight. Your summary should help understand not only what is happening at the moment but also what is likely to happen next with logical reasoning. The principle is safety first, so the prediction action should prioritize the driver's safety and secondly the pedestrians' safety. Be incredibly detailed. Output in the "next_action" field of the JSON format template.

                    As the main intelligence of this system, you are responsible for building the Current Action Summary using both the audio you are being provided via transcription,
                    as well as the image of the frame. Note: . Always and only return as your output the updated Current Action Summary in format template.
                    Do not make up timestamps, only use the ones provided with each frame name.

                    Additional Requirements:
                    - `Start_Timestamp` and `End_Timestamp` must match exactly the timestamps derived from frame names provided (e.g., "4.0s").
                    - `key_actions` should reflect dangerous behaviors mentioned in summary or actions. If none found, use "none".
                    - Avoid free-form descriptive text in `key_actions` and `next_action`.
                    - `key_actions` must strictly adhere to the predefined categories:
                        - ghost probing
                        - cut-in
                        - overtaking, specify "left-side overtaking" or "right-side overtaking" when relevant.

                        Exclude all other types of behaviors. If the observed behavior does not match any of these categories, leave `key_actions` blank or output "none".
                        For example:
                        - Correct: "key_actions": "ghost probing".
                        - Incorrect: "key_actions": "ghost probing, running across the road".

                    - All textual fields must be in English.
                    - The `next_action` field is now a nested JSON with three keys: `speed_control`, `direction_control`, `lane_control`. Each must choose one value from their respective sets.
                    - If there are multiple key actions, separate them by a comma, e.g. "ghost probing, cut-in".
                    - `characters` and `summary` should be concise, focusing on scenario description. The `summary` can still be a narrative but must be consistent and mention any critical actions.

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
                        - Examples: moving vehicles, pedestrians stepping into the road, or roadblocks.
                        - Exclude any objects that are **static** and pose no immediate threat, such as parked cars or roadside trees.
                    - Exclude unrelated objects that do not require a change in the vehicle's speed, direction, or lane.
                        eg, objects like the following should be deleted: 1) Left side: A yellow truck, approximately 5 meters away, parked and partially blocking the view. 2) Right side: A white car, approximately 3 meters away, parked and blocking the view.
                    - Ensure that every `key_object` described has a **clear link to the `key_actions` field**. If no clear link exists, remove the object.
                    - Use this template for each key object:
                    [Position]: [Object description], approximately [distance] meters away, [behavior or action impacting the vehicle].

                    **Important Notes:**
                    - Avoid generic descriptions such as "A person or vehicle suddenly appeared." Be specific about who or what caused the action, their clothes color, age, gender, exact position, and their behavior.
                    - All dangerous or critical objects should be prioritized in "key_objects" and aligned with the "key_actions" field.
                    - Make sure to use "{video_id}" as the value for the "video_id" field and "{segment_id_str}" for the "segment_id" field in your output.

                    Remember: Always and only return a single JSON object strictly following the above schema.

                    Your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON.
                    你现在是一名英文助手。无论我问什么问题，你都必须只用英文回答。请不要使用任何其他语言。You must always and only answer totally in **English** language!!! I can only read English language. Ensure all parts of the JSON output, including **summaries**, **actions**, **next_action**, and **THE WHOLE OUTPUT**, **MUST BE IN ENGLISH** If you answer ANY word in Chinese, you are fired immediately! Translate Chinese to English if there is Chinese in "next_action" field.

                    Example:
                        "video_id": "{video_id}",
                        "segment_id": "{segment_id_str}",
                        "Start_Timestamp": "8.4s",
                        "sentiment": "Negative",
                        "End_Timestamp": "12.0s",
                        "scene_theme": "Dramatic",
                        "characters": "Driver of a white sedan",
                        "summary": "In this segment, the vehicle is driving on a rural road. Ahead, a white sedan is parked on the roadside. The driver mentions that the tricycle ahead had yielded, but a collision still occurred.",
                        "actions": "The self-driving vehicle collided with the tricycle, possibly due to insufficient clearance on the right side. Over the past 4 seconds, the vehicle's speed gradually decreased, eventually reaching XX km/h in the last frame, maintaining a 30-meter distance from the vehicle ahead.",
                        "characters": "Driver of a white sedan",
                        "summary": "In this video, the vehicle is traveling on a rural road. Ahead, a white sedan is parked on the roadside. The driver mentions that the tricycle ahead had yielded but a collision still occurred.",
                        "actions": "The self-driving vehicle collided with the tricycle, possibly due to insufficient clearance on the right side. Over the past 4 seconds, the vehicle's speed gradually decreased, eventually reaching XX km/h in the final frame, maintaining a 30-meter distance from the vehicle ahead.",
                        "key_objects": "1) Front: A white sedan, relatively close, approximately 20 meters away, parked on the roadside, possibly about to start or turn suddenly. 2) Right: A red tricycle, approximately 0 meters away, has collided. It may stop for the driver to handle the situation or suddenly move forward.",
                        "key_actions": ""Select one or multiple categories from {{ghosting, cut-in, left/right-side vehicle overtaken, collision, none}}. If multiple categories apply, list them separatedly by commas. For cut-in, give detailed reasoning process!!!"",
                        "next_action": "{{
                                        "speed_control": "choose from accelerate, decelerate, rapid deceleration, slow steady driving, consistent speed driving, stop, wait, reverse",
                                        "direction_control": "choose from turn left, turn right, U-turn, steer to circumvent, keep direction, brake (for special circumstances), ...",
                                        "lane_control": "choose from change to left lane, change to right lane, slight left shift, slight right shift, maintain current lane, return to normal lane"
                        }}"

                        "Start_Timestamp": "xxxs",
                        "sentiment": "Negative",
                        "End_Timestamp": "xxxs",
                        "scene_theme": "Dramatic",
                        "characters": "car ahead",
                        "summary": "",
                        "actions": "",
                        "key_objects": "。",
                        "key_actions": "cut-in: an object from an adjacent lane moves into the self-vehicle's lane at a distance too close to the observer",
                        "next_action": ""

                    **Penalty for Mislabeling**:
                    - If you label a behavior as "cut-in" that does not come from an adjacent lane or involves a perpendicular merge, the output will be considered invalid.
                    - Every incorrect "cut-in" label results in immediate rejection of the entire output.
                    - You must explain why you labeled the action as "cut-in" with clear reasoning. If the reasoning is weak, the label will also be rejected.


                    Use these examples to understand how to analyze and analyze the new images. Now generate a similar JSON response for the following video analysis:

Audio Transcription: [No audio available for this analysis]'''
    
    def send_azure_openai_request(self, prompt, images):
        """发送Azure OpenAI请求 - 使用GPT-4.1，Temperature=0"""
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
            "max_tokens": 2000,
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
            
            # 获取原始VIP详细prompt
            prompt = self.get_original_vip_detailed_prompt(video_id)
            
            # 发送API请求
            self.logger.info(f"📤 发送GPT-4.1 API请求: {video_id}")
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
        self.logger.info("🚀 开始Run 11 Rerun2 原始VIP详细Prompt实验")
        
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
        filename = os.path.join(self.output_dir, f"run11_gpt41_rerun2_intermediate_{count}videos_{self.timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def save_final_results(self, count):
        """保存最终结果"""
        filename = os.path.join(self.output_dir, f"run11_gpt41_rerun2_final_results_{self.timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"💾 最终结果已保存: {filename}")

if __name__ == "__main__":
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run11_gpt41_ghost_probing_fewshot_100videos/run11-rerun2-original-vip-prompt"
    
    experiment = GPT41Run11Rerun2OriginalVIP(output_dir)
    results = experiment.run_experiment()
    
    print(f"\n🎉 Run 11 Rerun2 原始VIP详细Prompt实验完成！")
    print(f"📊 成功处理: {len(results)} 个视频")
    print(f"📁 结果目录: {output_dir}")