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

def analyze_video_with_gemini(video_id, segment_id, image_paths, audio_transcript, frame_interval, frames_per_interval):
    """使用Gemini分析视频片段"""
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
        
        # 构建完整的系统提示（与GPT-4完全相同）
        segment_id_str = segment_id if segment_id else f"segment_{1:03d}"
        
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
                    """
        
        # 替换占位符
        system_content = system_content.replace("{video_id}", video_id)
        system_content = system_content.replace("{segment_id_str}", segment_id_str)
        
        # 构建用户消息内容
        user_content = []
        
        # 添加音频转录信息
        if audio_transcript and audio_transcript.strip():
            user_content.append(f"Audio transcript: {audio_transcript}")
        else:
            user_content.append("Audio transcript: No audio available or transcription failed.")
        
        # 添加图像信息
        user_content.append(f"\nAnalyzing {len(images)} frames from video {video_id}:")
        for i, img_path in enumerate(image_paths):
            timestamp = os.path.basename(img_path).replace('.jpg', 's')
            user_content.append(f"Frame {i+1}: {timestamp}")
        
        user_message = "\n".join(user_content)
        
        # 配置Gemini模型
        generation_config = {
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_content
        )
        
        # 准备消息内容 - 文本 + 图像
        message_parts = [user_message] + images
        
        # 发送请求给Gemini
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                detailed_logger.info(f"尝试Gemini API调用，尝试次数: {retry_count + 1}/{max_retries}")
                
                response = model.generate_content(message_parts)
                
                if response and response.text:
                    detailed_logger.info("Gemini API调用成功")
                    return response.text
                else:
                    detailed_logger.warning("Gemini API返回空响应")
                    
            except Exception as e:
                retry_count += 1
                error_msg = f"Gemini API调用失败 (尝试 {retry_count}/{max_retries}): {str(e)}"
                detailed_logger.error(error_msg)
                
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # 指数退避
                    detailed_logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    detailed_logger.error("Gemini API调用达到最大重试次数")
                    return None
        
        return None
        
    except Exception as e:
        detailed_logger.error(f"分析视频片段时发生错误: {str(e)}")
        detailed_logger.error(traceback.format_exc())
        return None

def AnalyzeVideo(video_path, frame_interval, frames_per_interval, speed_mode=False, output_dir='.'):
    """分析视频的主函数"""
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取视频ID
        video_filename = os.path.basename(video_path)
        video_id = extract_video_id(video_path)
        
        detailed_logger.info(f"开始处理视频: {video_filename}, Video ID: {video_id}")
        
        # 创建临时帧目录
        frames_dir = get_process_frame_dir()
        os.makedirs(frames_dir, exist_ok=True)
        
        # 加载视频
        video_clip = VideoFileClip(video_path)
        video_duration = video_clip.duration
        
        detailed_logger.info(f"视频时长: {video_duration:.2f}秒")
        
        # 计算要处理的区间数量
        total_intervals = int(np.ceil(video_duration / frame_interval))
        detailed_logger.info(f"总共需要处理 {total_intervals} 个区间")
        
        # 存储所有结果
        all_results = []
        
        # 逐个处理每个区间
        for interval_idx in range(total_intervals):
            interval_start = interval_idx * frame_interval
            interval_end = min((interval_idx + 1) * frame_interval, video_duration)
            
            detailed_logger.info(f"处理区间 {interval_idx + 1}/{total_intervals}: {interval_start:.1f}s - {interval_end:.1f}s")
            
            # 提取这个区间的帧
            image_paths = []
            interval_duration = interval_end - interval_start
            
            if interval_duration > 0:
                # 计算要提取的帧数（不超过指定的frames_per_interval）
                frames_to_capture = min(frames_per_interval, max(1, int(interval_duration)))
                
                # 计算每帧的时间点
                if frames_to_capture == 1:
                    capture_points = [interval_start + interval_duration / 2]
                else:
                    capture_points = [
                        interval_start + (i * interval_duration / (frames_to_capture - 1)) 
                        for i in range(frames_to_capture)
                    ]
                
                # 提取帧
                for i, timestamp in enumerate(capture_points):
                    frame_filename = f"frame_{interval_idx}_{i}_{timestamp:.1f}.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    
                    try:
                        # 从视频中提取帧
                        frame_array = video_clip.get_frame(timestamp)
                        frame_image = Image.fromarray(frame_array.astype('uint8'))
                        frame_image.save(frame_path, 'JPEG', quality=95)
                        image_paths.append(frame_path)
                        
                    except Exception as e:
                        detailed_logger.warning(f"提取帧失败 {timestamp:.1f}s: {str(e)}")
            
            if not image_paths:
                detailed_logger.warning(f"区间 {interval_idx + 1} 没有成功提取任何帧，跳过")
                continue
            
            # 音频处理（简化版，不做实际转录）
            audio_transcript = "No audio transcription available for this analysis."
            
            # 使用Gemini分析这个区间
            segment_id = f"segment_{interval_idx + 1:03d}"
            analysis_result = analyze_video_with_gemini(
                video_id, segment_id, image_paths, audio_transcript, 
                frame_interval, frames_per_interval
            )
            
            if analysis_result:
                try:
                    # 处理Gemini返回的带有```json标记的响应
                    cleaned_result = analysis_result.strip()
                    if cleaned_result.startswith('```json'):
                        cleaned_result = cleaned_result[7:]  # 去掉 ```json
                    if cleaned_result.endswith('```'):
                        cleaned_result = cleaned_result[:-3]  # 去掉结尾的 ```
                    cleaned_result = cleaned_result.strip()
                    
                    # 尝试解析JSON响应
                    json_result = json.loads(cleaned_result)
                    all_results.append(json_result)
                    detailed_logger.info(f"区间 {interval_idx + 1} 分析完成")
                except json.JSONDecodeError as e:
                    detailed_logger.error(f"区间 {interval_idx + 1} JSON解析失败: {str(e)}")
                    detailed_logger.error(f"原始响应: {analysis_result[:500]}...")
            else:
                detailed_logger.error(f"区间 {interval_idx + 1} 分析失败")
            
            # 清理这个区间的帧文件
            for img_path in image_paths:
                try:
                    os.remove(img_path)
                except:
                    pass
        
        # 清理帧目录
        try:
            os.rmdir(frames_dir)
        except:
            pass
        
        # 关闭视频文件
        video_clip.close()
        
        # 保存结果
        if all_results:
            result_filename = f'actionSummary_{video_id}.json'
            result_path = os.path.join(output_dir, result_filename)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            detailed_logger.info(f"视频 {video_filename} 处理完成，结果保存到: {result_path}")
            return result_path
        else:
            detailed_logger.error(f"视频 {video_filename} 处理失败：没有生成任何有效结果")
            return None
            
    except Exception as e:
        detailed_logger.error(f"处理视频时发生错误: {str(e)}")
        detailed_logger.error(traceback.format_exc())
        return None

def process_video_wrapper(video_path, args):
    """视频处理的包装函数，用于多进程调用"""
    try:
        video_name = os.path.basename(video_path)
        # 检查是否已经处理过该视频
        video_id = extract_video_id(video_path)
        result_filename = f'actionSummary_{video_id}.json'
        result_path = os.path.join(args.output_dir, result_filename)
        
        if os.path.exists(result_path) and not args.no_skip and not args.retry_failed:
            print(f"视频 {video_name} 已经处理过，跳过")
            return (video_path, "skipped", None, 0)
        
        start_time = time.time()
        # 处理视频
        result = AnalyzeVideo(video_path, args.interval, args.frames, args.speed_mode, args.output_dir)
        processing_time = time.time() - start_time
        
        return (video_path, "success", result, processing_time)
    except Exception as e:
        error_msg = f"处理视频失败: {str(e)}"
        print(error_msg)
        if 'detailed_logger' in globals():
            detailed_logger.error(error_msg)
            detailed_logger.error(traceback.format_exc())
        return (video_path, "failed", str(e), 0)

def main():
    parser = argparse.ArgumentParser(description='Gemini批量视频分析工具 - DADA-100专用')
    parser.add_argument('--folder', type=str, default='result/DADA-100-videos', help='视频文件夹路径')
    parser.add_argument('--interval', type=int, default=10, help='每个间隔的秒数')
    parser.add_argument('--frames', type=int, default=10, help='每个间隔的帧数')
    parser.add_argument('--output-dir', type=str, default='result2/run12-gemini-1.5-flash', help='输出目录')
    parser.add_argument('--limit', type=int, help='限制处理的视频数量')
    parser.add_argument('--start-at', type=int, default=0, help='从第N个视频开始处理')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已处理的视频')
    parser.add_argument('--retry-failed', action='store_true', help='重试失败的视频')
    parser.add_argument('--processes', type=int, default=1, help='并行进程数')
    parser.add_argument('--speed-mode', action='store_true', help='快速模式')
    
    args = parser.parse_args()
    
    # 加载环境变量
    load_dotenv(dotenv_path=".env", override=True)
    
    # 获取Gemini API密钥（使用主key，已升级为付费账户）
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("错误: 请设置GEMINI_API_KEY环境变量")
        sys.exit(1)
    
    # 配置Gemini
    genai.configure(api_key=gemini_api_key)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.output_dir, 'processing_log.txt')
    global detailed_logger
    detailed_logger = setup_logger('detailed_logger', log_file, level=logging.INFO)
    
    detailed_logger.info(f"使用Gemini API Key: {gemini_api_key[:10]}...")
    
    # 获取视频文件列表
    if not os.path.exists(args.folder):
        detailed_logger.error(f"视频文件夹不存在: {args.folder}")
        sys.exit(1)
    
    video_extensions = ('.avi', '.mp4', '.mov', '.mkv')
    all_videos = []
    
    for file in os.listdir(args.folder):
        if file.lower().endswith(video_extensions):
            all_videos.append(os.path.join(args.folder, file))
    
    all_videos.sort()
    
    if not all_videos:
        detailed_logger.error(f"在文件夹 {args.folder} 中没有找到视频文件")
        sys.exit(1)
    
    # 应用start-at参数
    if args.start_at > 0:
        all_videos = all_videos[args.start_at:]
        detailed_logger.info(f"从第 {args.start_at + 1} 个视频开始处理")
    
    # 应用limit参数
    if args.limit:
        all_videos = all_videos[:args.limit]
        detailed_logger.info(f"限制处理 {args.limit} 个视频")
    
    detailed_logger.info(f"总共需要处理 {len(all_videos)} 个视频")
    
    # 处理视频
    success_count = 0
    failed_count = 0
    skipped_count = 0
    total_processing_time = 0
    
    # 创建处理函数
    process_func = partial(process_video_wrapper, args=args)
    
    # 使用进度条处理视频
    with tqdm.tqdm(total=len(all_videos), desc="处理视频", unit="video") as pbar:
        if args.processes > 1:
            # 多进程处理
            with multiprocessing.Pool(processes=args.processes) as pool:
                for result in pool.imap(process_func, all_videos):
                    video_path, status, result_data, processing_time = result
                    video_name = os.path.basename(video_path)
                    
                    if status == "success":
                        success_count += 1
                        pbar.set_postfix(success=success_count, failed=failed_count, skipped=skipped_count)
                        detailed_logger.info(f"✓ {video_name} 处理成功 ({processing_time:.1f}s)")
                    elif status == "skipped":
                        skipped_count += 1
                        pbar.set_postfix(success=success_count, failed=failed_count, skipped=skipped_count)
                    else:
                        failed_count += 1
                        pbar.set_postfix(success=success_count, failed=failed_count, skipped=skipped_count)
                        detailed_logger.error(f"✗ {video_name} 处理失败: {result_data}")
                    
                    total_processing_time += processing_time
                    pbar.update(1)
        else:
            # 单进程处理
            for video_path in all_videos:
                result = process_func(video_path)
                video_path, status, result_data, processing_time = result
                video_name = os.path.basename(video_path)
                
                if status == "success":
                    success_count += 1
                    pbar.set_postfix(success=success_count, failed=failed_count, skipped=skipped_count)
                    detailed_logger.info(f"✓ {video_name} 处理成功 ({processing_time:.1f}s)")
                elif status == "skipped":
                    skipped_count += 1
                    pbar.set_postfix(success=success_count, failed=failed_count, skipped=skipped_count)
                else:
                    failed_count += 1
                    pbar.set_postfix(success=success_count, failed=failed_count, skipped=skipped_count)
                    detailed_logger.error(f"✗ {video_name} 处理失败: {result_data}")
                
                total_processing_time += processing_time
                pbar.update(1)
    
    # 生成统计报告
    stats = {
        "total_videos": len(all_videos),
        "success_count": success_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "total_processing_time": total_processing_time,
        "average_processing_time": total_processing_time / max(1, success_count),
        "completion_time": datetime.datetime.now().isoformat(),
        "configuration": {
            "interval": args.interval,
            "frames": args.frames,
            "output_dir": args.output_dir,
            "processes": args.processes
        }
    }
    
    # 保存统计数据
    stats_file = os.path.join(args.output_dir, 'processing_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 输出最终统计
    detailed_logger.info("=" * 50)
    detailed_logger.info("处理完成统计:")
    detailed_logger.info(f"总视频数: {len(all_videos)}")
    detailed_logger.info(f"成功处理: {success_count}")
    detailed_logger.info(f"处理失败: {failed_count}")
    detailed_logger.info(f"跳过视频: {skipped_count}")
    detailed_logger.info(f"总处理时间: {total_processing_time:.1f}秒")
    if success_count > 0:
        detailed_logger.info(f"平均处理时间: {total_processing_time/success_count:.1f}秒/视频")
    detailed_logger.info("=" * 50)

if __name__ == "__main__":
    main()