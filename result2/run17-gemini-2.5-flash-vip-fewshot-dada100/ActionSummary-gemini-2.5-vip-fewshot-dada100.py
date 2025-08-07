#!/usr/bin/env python3
"""
Run 17: Gemini-2.5-Flash + VIP原始超详细prompt + Run 14 Few-shot Examples
DADA-100视频处理脚本

结合最详细的VIP prompt和经过验证的few-shot examples，
测试Gemini-2.5-Flash在100个视频上的性能表现。
"""

import re
import os
import json
import time
import logging
import traceback
import datetime
from functools import partial
import multiprocessing
import tqdm
import numpy as np
import cv2
import base64
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io
import argparse
import sys

# 禁用moviepy冗长的日志
logging.getLogger('moviepy').setLevel(logging.ERROR)

# 全局变量用于多进程存储当前进程编号
CURRENT_PROCESS_ID = 0

def get_process_frame_dir(process_id=None):
    if process_id is None:
        process_id = CURRENT_PROCESS_ID
    return f'frames_process_{process_id}'

def setup_logger(name, log_file, level=logging.INFO, append=True, console_output=True, format_str=None):
    """设置一个日志记录器，将信息记录到指定文件"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers = []
    
    mode = 'a' if append else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode, encoding='utf-8')
    
    if format_str is None:
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_str)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def extract_video_id(video_path):
    """从视频文件名中提取video_id"""
    video_filename = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_filename)[0]
    
    # DADA-2000 格式: images_1_001.avi -> images_1_001
    if video_name_without_ext.startswith('images_'):
        return video_name_without_ext
    else:
        return video_name_without_ext

def load_image_as_pil(image_path):
    """加载图像为PIL Image对象"""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        detailed_logger.error(f"加载图像失败 {image_path}: {str(e)}")
        return None

def get_vip_detailed_prompt(video_id, segment_id_str, frame_interval, frames_per_interval):
    """获取VIP原始超详细prompt"""
    system_content = f"""You are VideoAnalyzerGPT analyzing a series of SEQUENCIAL images taken from a video, where each image represents a consecutive moment in time.Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

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

                    **Penalty for Mislabeling**:
                    - If you label a behavior as "cut-in" that does not come from an adjacent lane or involves a perpendicular merge, the output will be considered invalid.
                    - Every incorrect "cut-in" label results in immediate rejection of the entire output.
                    - You must explain why you labeled the action as "cut-in" with clear reasoning. If the reasoning is weak, the label will also be rejected.

                    **Few-shot Learning Examples:**

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

                    Use these examples to understand how to analyze and analyze the new images. Now generate a similar JSON response for the following video analysis:
                    """

    # 替换占位符
    system_content = system_content.replace("{video_id}", video_id)
    system_content = system_content.replace("{segment_id_str}", segment_id_str)
    
    return system_content

def analyze_video_with_gemini(video_id, segment_id, image_paths, audio_transcript, frame_interval, frames_per_interval):
    """使用Gemini-2.5-Flash分析视频片段"""
    try:
        # 加载图像
        images = []
        valid_images = 0
        
        for image_path in image_paths:
            img = load_image_as_pil(image_path)
            if img is not None:
                images.append(img)
                valid_images += 1
            else:
                detailed_logger.warning(f"跳过无效图像: {image_path}")
        
        if valid_images == 0:
            detailed_logger.error("没有有效的图像用于分析")
            return None
        
        detailed_logger.info(f"成功加载 {valid_images}/{len(image_paths)} 个图像，准备进行分析")
        
        # 构建segment_id字符串
        segment_id_str = f"segment_{segment_id:03d}"
        
        # 获取VIP详细prompt + Few-shot examples
        system_prompt = get_vip_detailed_prompt(video_id, segment_id_str, frame_interval, frames_per_interval)
        
        # 构建用户消息
        user_message = f"Audio transcription: {audio_transcript}\n\nPlease analyze these {valid_images} frames from the video and provide the JSON analysis:"
        
        # 准备消息内容
        message_parts = [user_message] + images
        
        # 配置Gemini模型
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0.0,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
        )
        
        # 进行API调用，包含重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                detailed_logger.info(f"尝试Gemini API调用，尝试次数: {attempt + 1}/{max_retries}")
                
                # Gemini不支持system role，需要将system prompt包含在user message中
                full_user_message = f"{system_prompt}\n\n{user_message}"
                full_message_parts = [full_user_message] + images
                
                response = model.generate_content(full_message_parts)
                
                if response.text:
                    result_text = response.text.strip()
                    
                    # 清理Gemini API返回的markdown格式
                    if result_text.startswith('```json'):
                        result_text = result_text[7:]
                    if result_text.endswith('```'):
                        result_text = result_text[:-3]
                    
                    result_text = result_text.strip()
                    
                    # 尝试解析JSON
                    try:
                        result_json = json.loads(result_text)
                        detailed_logger.info("Gemini API调用成功")
                        return result_json
                    except json.JSONDecodeError as e:
                        detailed_logger.error(f"JSON解析失败: {str(e)}")
                        detailed_logger.error(f"原始响应: {result_text[:500]}...")
                        if attempt == max_retries - 1:
                            return None
                        continue
                else:
                    detailed_logger.error("Gemini API返回空响应")
                    
            except Exception as e:
                detailed_logger.error(f"Gemini API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    detailed_logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    detailed_logger.error("Gemini API调用达到最大重试次数")
                    return None
        
        return None
        
    except Exception as e:
        detailed_logger.error(f"分析视频时发生错误: {str(e)}")
        detailed_logger.error(traceback.format_exc())
        return None

def extract_frames(video_path, start_time, end_time, frames_per_interval, output_dir, process_id=None):
    """从视频中提取指定时间范围的帧"""
    try:
        if process_id is None:
            process_id = CURRENT_PROCESS_ID
        
        frame_dir = os.path.join(output_dir, get_process_frame_dir(process_id))
        
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        
        with VideoFileClip(video_path) as video:
            duration = end_time - start_time
            frame_files = []
            
            for i in range(frames_per_interval):
                if frames_per_interval == 1:
                    frame_time = start_time + duration / 2
                else:
                    frame_time = start_time + (duration * i / (frames_per_interval - 1))
                
                if frame_time > video.duration:
                    frame_time = video.duration - 0.1
                elif frame_time < 0:
                    frame_time = 0
                
                frame = video.get_frame(frame_time)
                frame_filename = f"frame_{frame_time:.1f}s.jpg"
                frame_path = os.path.join(frame_dir, frame_filename)
                
                frame_rgb = (frame * 255).astype(np.uint8)
                cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                frame_files.append(frame_path)
            
            return frame_files
    
    except Exception as e:
        detailed_logger.error(f"提取帧时发生错误: {str(e)}")
        return []

def process_single_video(video_path, interval, frames, output_dir, start_at_video=0):
    """处理单个视频文件"""
    global detailed_logger
    
    try:
        video_filename = os.path.basename(video_path)
        video_id = extract_video_id(video_path)
        
        detailed_logger.info(f"开始处理视频: {video_filename}, Video ID: {video_id}")
        
        # 检查输出文件是否已存在
        output_file = os.path.join(output_dir, f"actionSummary_{video_id}.json")
        if os.path.exists(output_file):
            detailed_logger.info(f"视频 {video_filename} 已经处理过，跳过")
            return video_id, "skipped", 0
        
        start_time = time.time()
        
        # 获取视频时长
        with VideoFileClip(video_path) as video:
            duration = video.duration
        
        detailed_logger.info(f"视频时长: {duration:.2f}秒")
        
        # 计算需要处理的区间数
        num_intervals = int(np.ceil(duration / interval))
        detailed_logger.info(f"总共需要处理 {num_intervals} 个区间")
        
        # 存储所有区间的分析结果
        all_segments = []
        
        # 处理每个时间区间
        for i in range(num_intervals):
            start_timestamp = i * interval
            end_timestamp = min((i + 1) * interval, duration)
            
            detailed_logger.info(f"处理区间 {i+1}/{num_intervals}: {start_timestamp:.1f}s - {end_timestamp:.1f}s")
            
            # 提取该区间的帧
            frame_files = extract_frames(
                video_path, start_timestamp, end_timestamp, 
                frames, output_dir, CURRENT_PROCESS_ID
            )
            
            if not frame_files:
                detailed_logger.error(f"区间 {i+1} 帧提取失败")
                continue
            
            # 音频转录（简化版本）
            audio_transcript = "No audio transcription available"
            
            # 调用Gemini分析
            segment_result = analyze_video_with_gemini(
                video_id, i, frame_files, audio_transcript, interval, frames
            )
            
            if segment_result:
                # 添加时间戳信息
                segment_result["Start_Timestamp"] = f"{start_timestamp:.1f}s"
                segment_result["End_Timestamp"] = f"{end_timestamp:.1f}s"
                all_segments.append(segment_result)
                detailed_logger.info(f"区间 {i+1} 分析完成")
            else:
                detailed_logger.error(f"区间 {i+1} 分析失败")
                return video_id, "failed", time.time() - start_time
            
            # 清理帧文件
            for frame_file in frame_files:
                try:
                    os.remove(frame_file)
                except:
                    pass
        
        if not all_segments:
            detailed_logger.error(f"视频 {video_filename} 处理失败：没有生成任何有效结果")
            return video_id, "failed", time.time() - start_time
        
        # 保存结果到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_segments, f, indent=2, ensure_ascii=False)
        
        processing_time = time.time() - start_time
        detailed_logger.info(f"视频 {video_filename} 处理完成，结果保存到: {output_file}")
        detailed_logger.info(f"✓ {video_filename} 处理成功 ({processing_time:.1f}s)")
        
        return video_id, "success", processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        detailed_logger.error(f"处理视频 {video_path} 时发生错误: {str(e)}")
        detailed_logger.error(traceback.format_exc())
        return extract_video_id(video_path), "failed", processing_time

def main():
    global detailed_logger, CURRENT_PROCESS_ID
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Run 17: Gemini-2.5-Flash + VIP详细prompt + Few-shot处理DADA-100视频')
    parser.add_argument('--folder', required=True, help='视频文件夹路径')
    parser.add_argument('--interval', type=int, default=10, help='时间间隔（秒）')
    parser.add_argument('--frames', type=int, default=10, help='每个间隔提取的帧数')
    parser.add_argument('--start-at', type=int, default=0, help='从第N个视频开始处理')
    parser.add_argument('--limit', type=int, help='处理视频的最大数量')
    parser.add_argument('--processes', type=int, default=1, help='并行进程数')
    
    args = parser.parse_args()
    
    # 设置工作目录
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run17-gemini-2.5-flash-vip-fewshot-dada100"
    os.chdir(output_dir)
    
    # 设置日志
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"processing_log_{timestamp}.txt"
    detailed_logger = setup_logger('detailed', log_file, level=logging.INFO, console_output=True)
    
    # 加载环境变量
    load_dotenv('/Users/wanmeng/repository/GPT4Video-cobra-auto/.env')
    
    # 配置Gemini API
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("错误: 请设置GEMINI_API_KEY环境变量")
        return
    
    detailed_logger.info(f"使用Gemini API Key: {gemini_api_key[:10]}...")
    genai.configure(api_key=gemini_api_key)
    
    # 记录配置信息
    detailed_logger.info("=" * 50)
    detailed_logger.info("Run 17: Gemini-2.5-Flash + VIP详细prompt + Few-shot Examples")
    detailed_logger.info("=" * 50)
    detailed_logger.info(f"视频文件夹: {args.folder}")
    detailed_logger.info(f"时间间隔: {args.interval}秒")
    detailed_logger.info(f"每间隔帧数: {args.frames}帧")
    detailed_logger.info(f"从第 {args.start_at + 1} 个视频开始处理")
    if args.limit:
        detailed_logger.info(f"限制处理 {args.limit} 个视频")
    detailed_logger.info(f"输出目录: {output_dir}")
    
    # 获取视频文件列表
    video_files = []
    for ext in ['*.avi', '*.mp4', '*.mov']:
        video_files.extend(glob.glob(os.path.join(args.folder, ext)))
    
    video_files.sort()
    detailed_logger.info(f"发现 {len(video_files)} 个视频文件")
    
    # 应用起始位置和限制
    if args.start_at > 0:
        video_files = video_files[args.start_at:]
        detailed_logger.info(f"从第 {args.start_at + 1} 个视频开始处理")
    
    if args.limit:
        video_files = video_files[:args.limit]
        detailed_logger.info(f"限制处理 {len(video_files)} 个视频")
    
    detailed_logger.info(f"总共需要处理 {len(video_files)} 个视频")
    
    # 处理视频
    results = []
    start_time = time.time()
    
    # 使用tqdm显示进度
    with tqdm.tqdm(total=len(video_files), desc="处理视频", unit="video") as pbar:
        for video_path in video_files:
            result = process_single_video(video_path, args.interval, args.frames, output_dir, args.start_at)
            results.append(result)
            
            # 更新进度条
            video_id, status, processing_time = result
            success_count = sum(1 for r in results if r[1] == "success")
            failed_count = sum(1 for r in results if r[1] == "failed") 
            skipped_count = sum(1 for r in results if r[1] == "skipped")
            
            pbar.set_postfix({
                'success': success_count,
                'failed': failed_count, 
                'skipped': skipped_count
            })
            pbar.update(1)
    
    # 计算统计信息
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r[1] == "success")
    failed_count = sum(1 for r in results if r[1] == "failed")
    skipped_count = sum(1 for r in results if r[1] == "skipped")
    total_processing_time = sum(r[2] for r in results if r[1] == "success")
    avg_processing_time = total_processing_time / success_count if success_count > 0 else 0
    
    # 输出统计信息
    detailed_logger.info("=" * 50)
    detailed_logger.info("处理完成统计:")
    detailed_logger.info(f"总视频数: {len(video_files)}")
    detailed_logger.info(f"成功处理: {success_count}")
    detailed_logger.info(f"处理失败: {failed_count}")
    detailed_logger.info(f"跳过视频: {skipped_count}")
    detailed_logger.info(f"总处理时间: {total_processing_time:.1f}秒")
    detailed_logger.info(f"平均处理时间: {avg_processing_time:.1f}秒/视频")
    detailed_logger.info("=" * 50)
    
    # 保存统计信息
    stats = {
        "total_videos": len(video_files),
        "success_count": success_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "total_processing_time": total_processing_time,
        "average_processing_time": avg_processing_time,
        "completion_time": datetime.datetime.now().isoformat(),
        "configuration": {
            "interval": args.interval,
            "frames": args.frames,
            "output_dir": "run17-gemini-2.5-flash-vip-fewshot-dada100",
            "processes": args.processes
        }
    }
    
    with open("processing_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    detailed_logger.info(f"统计信息已保存到 processing_stats.json")
    detailed_logger.info("Run 17 处理完成！")

if __name__ == "__main__":
    import glob
    main()