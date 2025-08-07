#!/usr/bin/env python3
"""
Run 14: Gemini 2.0 Flash + VIP Prompt + Few-shot Examples (100 Videos)
基于Run 13，添加Run 8的Few-shot Examples以提升性能
最优配置：VIP Professional Prompt + Few-shot Learning + Temperature=0
"""

import os
import sys
import json
import time
import datetime
import traceback
import logging
from pathlib import Path
import base64
import cv2
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append('/Users/wanmeng/repository/GPT4Video-cobra-auto')
import video_utilities as vu

# 加载环境变量
load_dotenv(dotenv_path="/Users/wanmeng/repository/GPT4Video-cobra-auto/.env", override=True)

class Run14GeminiAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.setup_gemini()
        self.setup_directories()
        
    def setup_logging(self):
        """设置日志记录"""
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"run14_gemini_fewshot_{self.timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Run 14 Gemini 2.0 Flash + Few-shot Analysis Started - {self.timestamp}")
        
    def setup_gemini(self):
        """设置Gemini API - 支持双密钥策略"""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.api_key_2 = os.environ.get("GEMINI_API_KEY_2")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.current_api_key = self.api_key
        self.logger.info(f"Gemini 2.0 Flash model initialized: {self.api_key[:10]}...")
        if self.api_key_2:
            self.logger.info(f"Backup API key available: {self.api_key_2[:10]}...")
        
    def switch_api_key(self):
        """切换到备用API密钥"""
        if self.api_key_2 and self.current_api_key == self.api_key:
            self.current_api_key = self.api_key_2
            genai.configure(api_key=self.api_key_2)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.logger.info("Switched to backup API key")
            return True
        return False
        
    def setup_directories(self):
        """设置目录路径"""
        self.project_root = Path("/Users/wanmeng/repository/GPT4Video-cobra-auto")
        self.dada_100_dir = self.project_root / "result" / "DADA-100-videos"
        self.output_dir = Path(__file__).parent
        self.frames_temp_dir = self.output_dir / "frames_temp"
        
        # 创建临时帧目录
        self.frames_temp_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"DADA-100 videos directory: {self.dada_100_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def get_vip_prompt_with_fewshot(self, video_id, segment_id_str):
        """获取VIP prompt + Few-shot Examples的完整版本"""
        system_content = f"""
        You are an expert driver assistance AI system specialized in analyzing driving scenarios for autonomous vehicles. 
        Your primary task is to detect and classify dangerous driving behaviors, particularly focusing on "ghost probing" incidents.

        **Critical Definitions:**

        ### Ghost Probing (鬼探头)
        A "ghost probing" incident occurs when a vehicle or pedestrian suddenly appears from behind a physical obstruction (such as a parked car, building, or visual barrier) with minimal warning time for the driver. This creates an extremely dangerous situation requiring immediate reaction.

        Key characteristics:
        1. The object/person was **not visible** before appearing due to physical obstruction
        2. The appearance is **sudden and unexpected**
        3. Creates **immediate danger** requiring emergency response
        4. Gives driver **minimal reaction time**

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

        ### Cut-in vs Ghost Probing Distinction
        Typically **within same-direction traffic flow**, a cut-in happens when a vehicle deliberately forces its way in front of another vehicle's traffic lane from the **adjacent lane**, occupying another driver's lane space. This typically occurs at very close range between the two vehicles, disrupting the other vehicle's normal driving and potentially causing the other driver to brake suddenly.

        Characteristics:
        A cut-in is defined only when a vehicle merges into the current lane from an adjacent side lane.
        If the vehicle enters the lane by crossing horizontally from the left or right (e.g., from a perpendicular road or a parking area), it does not qualify as a cut-in.

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

        Your angle appears to watch video frames recorded from a surveillance camera in a car. Your role should focus on detecting and predicting dangerous actions in a \"Ghosting\" manner
        where pedestrians or vehicles in the scene might suddenly appear in front of the current car. This could happen if a person or vehicle suddenly emerges from behind an obstacle in the driver's view.
        This behavior is extremely dangerous because it gives the driver very little time to react.
        Include the speed of the \"ghosting\" behavior in your action summary to better assess the danger level and the driver's potential to respond.

        Provide detailed description of both people's and vehicles' behavior and potential dangerous actions that could lead to collisions. Describe how you think the individual or vehicle could crash into the car, and explain your deduction process. Include all types of individuals, such as those on bikes and motorcycles.
        Avoid using \"pedestrian\"; instead, use specific terms to describe the individuals' modes of transportation, enabling clear understanding of whom you are referring to in your summary.
        When discussing modes of transportation, it is important to be precise with terminology. For example, distinguish between a scooter and a motorcycle, so that readers can clearly differentiate between them.
        Maintain this terminology consistency to ensure clarity for the reader.
        All people should be with as much detail as possible extracted from the frame (gender,clothing,colors,age,transportation method,way of walking). Be incredibly detailed. Output in the \"summary\" field of the JSON format template.

        **Task 2: Explain Current Driving Actions**
        Analyze the current video frames to extract actions. Describe not only the actions themselves but also provide detailed reasoning for why the vehicle is taking these actions, such as changes in speed and direction. Focus solely on the reasoning for the vehicle's actions, excluding any descriptions of pedestrian behavior. Explain why the driver is driving at a certain speed, making turns, or stopping. Your goal is to provide a comprehensive understanding of the vehicle's behavior based on the visual data. Output in the \"actions\" field of the JSON format template.

        **Task 3: Predict Next Driving Action**
        Understand the current road conditions, the driving behavior, and to predict the next driving action. Analyze the video and audio to provide a comprehensive summary of the road conditions, including weather, traffic density, road obstacles, and traffic light if visible. Predict the next driving action based on two dimensions, one is driving speed control, such as accelerating, braking, turning, or stopping, the other one is to predict the next lane control, such as change to left lane, change to right lane, keep left in this lane, keep right in this lane, keep straight. Your summary should help understand not only what is happening at the moment but also what is likely to happen next with logical reasoning. The principle is safety first, so the prediction action should prioritize the driver's safety and secondly the pedestrians' safety. Be incredibly detailed. Output in the \"next_action\" field of the JSON format template.

        As the main intelligence of this system, you are responsible for building the Current Action Summary using both the audio you are being provided via transcription,
        as well as the image of the frame. Note: . Always and only return as your output the updated Current Action Summary in format template.
        Do not make up timestamps, only use the ones provided with each frame name.

        Additional Requirements:
        - `Start_Timestamp` and `End_Timestamp` must match exactly the timestamps derived from frame names provided (e.g., \"4.0s\").
        - `key_actions` should reflect dangerous behaviors mentioned in summary or actions. If none found, use \"none\".
        - Avoid free-form descriptive text in `key_actions` and `next_action`.
        - `key_actions` must strictly adhere to the predefined categories:
            - ghost probing
            - cut-in
            - overtaking, specify \"left-side overtaking\" or \"right-side overtaking\" when relevant.

            Exclude all other types of behaviors. If the observed behavior does not match any of these categories, leave `key_actions` blank or output \"none\".
            For example:
            - Correct: \"key_actions\": \"ghost probing\".
            - Incorrect: \"key_actions\": \"ghost probing, running across the road\".

        - All textual fields must be in English.
        - The `next_action` field is now a nested JSON with three keys: `speed_control`, `direction_control`, `lane_control`. Each must choose one value from their respective sets.
        - If there are multiple key actions, separate them by a comma, e.g. \"ghost probing, cut-in\".
        - `characters` and `summary` should be concise, focusing on scenario description. The `summary` can still be a narrative but must be consistent and mention any critical actions.

        **Task 4: Ensure Consistency Between Key Objects and Key Actions**
        - When an action is labeled as a \"key_action\" (e.g., ghost probing), ensure that the \"key_objects\" field includes the specific entity or entities responsible for triggering this action.
        - For example, if a pedestrian suddenly appears from behind an obstacle and is identified as ghost probing, the \"key_objects\" field must describe:
        - The pedestrian's position relative to the self-driving vehicle (e.g., left side, right side, etc.).
        - The pedestrian's behavior leading to the key action (e.g., moving suddenly from behind a parked truck).
        - The potential impact on the vehicle (e.g., causing the vehicle to decelerate or stop).
        - Each key object description should include:
        - Relative position (e.g., left, right, front).
        - Distance from the vehicle in meters.
        - Movement direction or behavior (e.g., approaching, crossing, accelerating).
        - The relationship to the \"key_action\" it caused.
        - Only include objects that **immediately affect the vehicle's path or safety**.
            - Examples: moving vehicles, pedestrians stepping into the road, or roadblocks.
            - Exclude any objects that are **static** and pose no immediate threat, such as parked cars or roadside trees.
        - Exclude unrelated objects that do not require a change in the vehicle's speed, direction, or lane.
        - Ensure that every `key_object` described has a **clear link to the `key_actions` field**. If no clear link exists, remove the object.
        - Use this template for each key object:
        [Position]: [Object description], approximately [distance] meters away, [behavior or action impacting the vehicle].

        **Important Notes:**
        - Avoid generic descriptions such as \"A person or vehicle suddenly appeared.\" Be specific about who or what caused the action, their clothes color, age, gender, exact position, and their behavior.
        - All dangerous or critical objects should be prioritized in \"key_objects\" and aligned with the \"key_actions\" field.
        - Make sure to use \"{video_id}\" as the value for the \"video_id\" field and \"{segment_id_str}\" for the \"segment_id\" field in your output.

        Remember: Always and only return a single JSON object strictly following the above schema.

        Your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON.
        You must always and only answer totally in **English** language!!! I can only read English language. Ensure all parts of the JSON output, including **summaries**, **actions**, **next_action**, and **THE WHOLE OUTPUT**, **MUST BE IN ENGLISH** If you answer ANY word in Chinese, you are fired immediately! Translate Chinese to English if there is Chinese in \"next_action\" field.

        **Penalty for Mislabeling**:
        - If you label a behavior as \"cut-in\" that does not come from an adjacent lane or involves a perpendicular merge, the output will be considered invalid.
        - Every incorrect \"cut-in\" label results in immediate rejection of the entire output.
        - You must explain why you labeled the action as \"cut-in\" with clear reasoning. If the reasoning is weak, the label will also be rejected.

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
            "segment_id": "{segment_id_str}",
            "Start_Timestamp": "0.0s",
            "End_Timestamp": "10.0s",
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

        Use these examples to understand how to analyze and analyze the new images. Now generate a similar JSON response for the following video analysis:
        """

        # 替换占位符
        system_content = system_content.replace("{video_id}", video_id)
        system_content = system_content.replace("{segment_id_str}", segment_id_str)
        
        return system_content
        
    def extract_frames_from_video(self, video_path, interval=10, max_frames=10):
        """从视频中提取帧 - 基于VIP脚本的参数"""
        try:
            frames = []
            video_clip = VideoFileClip(str(video_path))
            duration = video_clip.duration
            
            # 计算帧提取时间点
            for i in range(0, int(duration), interval):
                end_time = min(i + interval, duration)
                segment_duration = end_time - i
                
                # 在每个interval内均匀提取帧
                times_in_segment = []
                if segment_duration > 0:
                    step = segment_duration / max_frames
                    for j in range(max_frames):
                        frame_time = i + j * step
                        if frame_time < duration:
                            times_in_segment.append(frame_time)
                
                # 提取帧
                for frame_time in times_in_segment:
                    try:
                        frame = video_clip.get_frame(frame_time)
                        # 转换为OpenCV格式
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # 保存临时帧文件
                        frame_filename = f"frame_{i}_{len(frames)}_{frame_time:.1f}s.jpg"
                        frame_path = self.frames_temp_dir / frame_filename
                        cv2.imwrite(str(frame_path), frame_bgr)
                        
                        # 编码为base64
                        _, buffer = cv2.imencode('.jpg', frame_bgr)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        frames.append({
                            'timestamp': frame_time,
                            'filename': frame_filename,
                            'base64': frame_base64
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to extract frame at {frame_time}s: {e}")
                        continue
            
            video_clip.close()
            return frames
            
        except Exception as e:
            self.logger.error(f"Failed to extract frames from {video_path}: {e}")
            return []
    
    def analyze_video_with_gemini(self, video_path):
        """使用Gemini 2.0 Flash分析视频 - VIP + Few-shot版本"""
        try:
            video_name = video_path.stem
            self.logger.info(f"Analyzing video: {video_name}")
            
            # 提取帧
            frames = self.extract_frames_from_video(video_path, interval=10, max_frames=10)
            if not frames:
                self.logger.error(f"No frames extracted from {video_name}")
                return None
            
            # 准备prompt
            video_id = video_name
            segment_id_str = "segment_1"
            system_prompt = self.get_vip_prompt_with_fewshot(video_id, segment_id_str)
            
            # 准备用户消息内容
            user_content = f"""Analyze the following video frames from {video_id}:

Frame information:
"""
            
            # 添加帧信息
            for i, frame in enumerate(frames):
                user_content += f"Frame {i+1}: {frame['filename']} (timestamp: {frame['timestamp']:.1f}s)\n"
            
            user_content += """
Please analyze these frames and return a JSON response following the exact format specified in the system prompt.
Focus on detecting any ghost probing incidents or other dangerous behaviors.
Use the Few-shot examples as guidance for the level of detail and accuracy expected.
"""
            
            # 准备图像内容用于Gemini
            image_parts = []
            for frame in frames:
                image_parts.append({
                    'mime_type': 'image/jpeg',
                    'data': base64.b64decode(frame['base64'])
                })
            
            # 构建完整的prompt
            full_prompt = [system_prompt + "\n\n" + user_content] + image_parts
            
            # 调用Gemini API with enhanced retry and dual key support
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"API call attempt {attempt+1}/{max_retries} for {video_name}")
                    response = self.model.generate_content(
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0,
                            max_output_tokens=4000,
                        )
                    )
                    break
                except Exception as e:
                    error_str = str(e)
                    if "RATE_LIMIT_EXCEEDED" in error_str or "429" in error_str:
                        # 尝试切换API密钥
                        if attempt == 2 and self.switch_api_key():
                            self.logger.info("Switched to backup API key due to rate limit")
                            continue
                        
                        wait_time = (attempt + 1) * 45
                        self.logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    elif "SAFETY" in error_str:
                        self.logger.error(f"Safety filter triggered for {video_name}: {e}")
                        return None
                    else:
                        self.logger.error(f"API error for {video_name} attempt {attempt+1}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(10)
                            continue
                        else:
                            raise e
            else:
                self.logger.error(f"Max retries exceeded for {video_name}")
                return None
            
            # 解析响应
            if response.text:
                try:
                    # 清理响应文本，移除markdown代码块
                    clean_text = response.text.strip()
                    if clean_text.startswith('```json'):
                        clean_text = clean_text[7:]  # 移除 ```json
                    if clean_text.endswith('```'):
                        clean_text = clean_text[:-3]  # 移除 ```
                    clean_text = clean_text.strip()
                    
                    # 尝试解析JSON
                    result = json.loads(clean_text)
                    self.logger.info(f"Successfully analyzed {video_name}")
                    return result
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON response for {video_name}: {e}")
                    self.logger.debug(f"Raw response: {response.text}")
                    self.logger.debug(f"Cleaned response: {clean_text[:500]}...")
                    return None
            else:
                self.logger.error(f"Empty response for {video_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error analyzing {video_path}: {e}")
            self.logger.error(traceback.format_exc())
            return None
        finally:
            # 清理临时帧文件
            try:
                for frame_file in self.frames_temp_dir.glob("frame_*.jpg"):
                    frame_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to clean temp frames: {e}")
    
    def get_video_files(self):
        """获取DADA-100视频文件列表"""
        video_files = []
        for video_file in self.dada_100_dir.glob("images_*.avi"):
            video_files.append(video_file)
        
        video_files.sort()
        self.logger.info(f"Found {len(video_files)} video files to process")
        return video_files
    
    def run_analysis(self):
        """运行完整分析"""
        try:
            # 获取视频文件列表
            video_files = self.get_video_files()
            
            if not video_files:
                self.logger.error("No video files found!")
                return
            
            # 分析统计
            results = {}
            processed_count = 0
            failed_count = 0
            
            self.logger.info(f"Starting analysis of {len(video_files)} videos")
            
            # 处理每个视频
            with tqdm(video_files, desc="Analyzing videos") as pbar:
                for video_path in pbar:
                    video_name = video_path.stem
                    pbar.set_description(f"Processing {video_name}")
                    
                    # 检查是否已处理
                    result_file = self.output_dir / f"actionSummary_{video_name}.json"
                    if result_file.exists():
                        self.logger.info(f"Skipping {video_name} - already processed")
                        processed_count += 1
                        continue
                    
                    # 分析视频
                    result = self.analyze_video_with_gemini(video_path)
                    
                    if result:
                        # 保存结果
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        
                        results[video_name] = result
                        processed_count += 1
                        self.logger.info(f"Saved result for {video_name}")
                    else:
                        failed_count += 1
                        self.logger.error(f"Failed to analyze {video_name}")
                    
                    # 添加延迟避免API限制
                    time.sleep(3)
            
            # 保存分析汇总结果
            summary = {
                'experiment_info': {
                    'run_id': 'Run 14',
                    'timestamp': self.timestamp,
                    'model': 'gemini-2.0-flash-exp',
                    'prompt_version': 'VIP Professional + Few-shot Examples',
                    'total_videos': len(video_files),
                    'processed_videos': processed_count,
                    'failed_videos': failed_count,
                    'processing_parameters': {
                        'interval': 10,
                        'max_frames': 10,
                        'temperature': 0,
                        'max_output_tokens': 4000
                    },
                    'key_features': [
                        'VIP Professional Prompt',
                        'Few-shot Learning Examples',
                        'Dual API Key Strategy',
                        'Enhanced Error Handling',
                        'Temperature=0 for Consistency'
                    ]
                },
                'results': results
            }
            
            summary_file = self.output_dir / f"run14_analysis_summary_{self.timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Analysis completed! Processed: {processed_count}, Failed: {failed_count}")
            self.logger.info(f"Summary saved to: {summary_file}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in run_analysis: {e}")
            self.logger.error(traceback.format_exc())
            return None

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Run 14: Gemini 2.0 Flash + VIP + Few-shot (100 Videos)")
    print("=" * 60)
    
    try:
        analyzer = Run14GeminiAnalyzer()
        result = analyzer.run_analysis()
        
        if result:
            print(f"\n✅ Analysis completed successfully!")
            print(f"📊 Processed: {result['experiment_info']['processed_videos']} videos")
            print(f"❌ Failed: {result['experiment_info']['failed_videos']} videos")
        else:
            print("\n❌ Analysis failed!")
            
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()