#!/usr/bin/env python3
"""
DriveLM风格GPT-4.1脚本 - 使用Graph VQA方法进行Ghost Probing检测
基于DriveLM的Graph Visual Question Answering方法论
"""

import re
from functools import partial
import multiprocessing
import datetime
import traceback
import tqdm
import numpy as np
from jinja2 import Environment, FileSystemLoader
import video_utilities as vu
from dotenv import load_dotenv
from functools import wraps
import time
import cv2
import os
import base64
import requests
from moviepy.editor import VideoFileClip
import logging
import json
import sys
import openai
import threading
from retrying import retry
logging.getLogger('moviepy').setLevel(logging.ERROR)

# 全局变量用于多进程存储当前进程编号
CURRENT_PROCESS_ID = 0

# 获取当前进程专用的帧目录
def get_process_frame_dir(process_id=None):
    if process_id is None:
        process_id = CURRENT_PROCESS_ID
    return f'frames_process_{process_id}'

# 初始化日志记录器
def initialize_logger(log_level='INFO'):
    logger = logging.getLogger('video_processor')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

# 设置详细日志记录器
detailed_logger = initialize_logger('INFO')

# 加载环境变量
load_dotenv()

# 配置Azure OpenAI API
azure_speech_key = os.environ["AZURE_SPEECH_KEY"]
azure_whisper_key = os.environ["AZURE_WHISPER_KEY"]
azure_whisper_deployment = os.environ["AZURE_WHISPER_DEPLOYMENT"]
azure_whisper_endpoint = os.environ["AZURE_WHISPER_ENDPOINT"]

# Audio API type (OpenAI, Azure)*
audio_api_type = os.environ["AUDIO_API_TYPE"]

# GPT4 vision APi type (OpenAI, Azure)*
vision_api_type = os.environ["VISION_API_TYPE"]

# OpenAI API Key*
openai_api_key = os.environ["OPENAI_API_KEY"]

# GPT-4.1 Azure vision API Deployment Name*
vision_deployment = os.environ.get("VISION_ENDPOINT_4.1", "gpt-4.1")

# GPT endpoint  
vision_endpoint = os.environ["VISION_ENDPOINT"]

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            detailed_logger.info(f"函数 {func.__name__} 执行完成，耗时: {execution_time:.2f}秒")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            detailed_logger.error(f"函数 {func.__name__} 执行失败，耗时: {execution_time:.2f}秒，错误: {str(e)}")
            raise
    return wrapper

class Spinner:
    def __init__(self, message="处理中..."):
        self.message = message
        self.spinner_chars = "|/-\\"
        self.stop_spinner = False
        self.thread = None
        self.logger = detailed_logger

    def spin(self):
        i = 0
        while not self.stop_spinner:
            sys.stdout.write(
                f'\r{self.message} {self.spinner_chars[i % len(self.spinner_chars)]}')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        self.stop_spinner = False
        self.thread = threading.Thread(target=self.spin)
        self.thread.daemon = True
        self.thread.start()
        if self.logger:
            self.logger.debug(f"Spinner thread started for: {self.message}")

    def stop(self):
        self.stop_spinner = True
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            sys.stdout.write(
                '\r' + ' '*(len(self.message)+2) + '\r')
            sys.stdout.flush()
            if self.logger:
                self.logger.debug(f"Spinner stopped: {self.message}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error stopping spinner: {str(e)}")
            try:
                sys.stdout.write('\r' + ' '*(len(self.message)+2) + '\r')
                sys.stdout.flush()
            except:
                pass

chapter_summary = {}
miss_arr = []

def extract_video_id(filename, logger=None):
    if logger:
        logger.debug(f"提取视频ID，文件名: {filename}")

    try:
        if not filename or not isinstance(filename, str):
            error_msg = f"无效的文件名: {filename}"
            if logger:
                logger.error(error_msg)
            return f"unknown_{int(time.time())}"

        dada_match = re.match(r"images_(\d+)_(\d+)\.avi", filename)
        if dada_match:
            video_id = f"dada_{dada_match.group(1)}_{dada_match.group(2)}"
            if logger:
                logger.debug(f"匹配DADA格式成功: {video_id}")
            return video_id

        num_prefix_match = re.match(r"(\d+)_", filename)
        if num_prefix_match:
            video_id = f"vid_{num_prefix_match.group(1)}"
            if logger:
                logger.debug(f"匹配数字前缀格式成功: {video_id}")
            return video_id

        base_name = os.path.splitext(filename)[0]
        video_id = re.sub(r'[^\w-]', '_', base_name)
        if logger:
            logger.debug(f"使用通用备选方案: {video_id}")
        return video_id
    except Exception as e:
        error_msg = f"提取视频ID时出错: {str(e)}, 回退到安全ID"
        if logger:
            logger.error(error_msg)
            logger.error(traceback.format_exc())

        safe_id = f"unknown_{int(time.time())}_{hash(filename) % 10000 if filename else 0}"
        return safe_id

@log_execution_time
def AnalyzeVideo(vp, fi, fpi, speed_mode=False, output_dir='result/drivelm_comparison/drivelm_gpt41_results'):
    video_filename = os.path.basename(vp)
    video_id = extract_video_id(video_filename)
    print(f"处理视频: {video_filename}, 视频ID: {video_id}")

    global CURRENT_PROCESS_ID
    if multiprocessing.current_process().name != 'MainProcess':
        frames_dir = get_process_frame_dir()
    else:
        frames_dir = 'frames'

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_clip = VideoFileClip(vp)
    duration = video_clip.duration
    detailed_logger.info(f"视频时长: {duration:.2f}秒")

    total_intervals = int(duration / fi)
    if duration % fi > 0:
        total_intervals += 1

    detailed_logger.info(f"将创建 {total_intervals} 个间隔，每间隔 {fi} 秒，每间隔 {fpi} 帧")

    progress_bar = tqdm.tqdm(total=total_intervals, desc="处理视频间隔", unit="interval")
    all_segments = []

    for interval_index in range(total_intervals):
        segment_id = f"segment_{interval_index:03d}"
        start_time = interval_index * fi
        end_time = min(start_time + fi, duration)
        
        detailed_logger.info(f"处理间隔 {interval_index + 1}/{total_intervals}: {start_time:.1f}s - {end_time:.1f}s")

        packet = []
        for frame_index in range(fpi):
            frame_time = start_time + (frame_index * (end_time - start_time) / fpi)
            if frame_time >= duration:
                break
            
            frame_filename = f"{frames_dir}/frame_at_{frame_time:.1f}s.jpg"
            
            frame = video_clip.get_frame(frame_time)
            cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            packet.append(frame_filename)
            detailed_logger.debug(f"提取帧: {frame_filename}")

        if not packet:
            detailed_logger.warning(f"间隔 {segment_id} 没有提取到帧，跳过")
            continue

        current_transcription = ""
        if video_clip.audio is not None:
            spinner = Spinner(f"正在转录间隔 {segment_id} 的音频...")
            spinner.start()
            
            try:
                audio_filename = f"{frames_dir}/audio_segment_{segment_id}.wav"
                audio_clip = video_clip.subclip(start_time, end_time)
                audio_clip.audio.write_audiofile(audio_filename, verbose=False, logger=None)
                
                tscribe = transcribe_audio(audio_filename, azure_whisper_key, azure_whisper_deployment, azure_whisper_endpoint)
                current_transcription = tscribe
                
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
                    
            except Exception as e:
                detailed_logger.error(f"音频转录失败: {str(e)}")
                current_transcription = ""
            finally:
                spinner.stop()
        else:
            print("视频没有音轨，跳过音频提取和转录")

        spinner = Spinner(f"正在分析间隔 {segment_id} 的 {len(packet)} 帧图像...")
        spinner.start()

        vision_response = gpt41_vision_analysis_balanced(
            packet, openai_api_key, "", current_transcription, video_id, segment_id, speed_mode, start_time, end_time)

        spinner.stop()

        if vision_response == -1:
            print(f"警告: 间隔 {segment_id} 的视觉分析失败，跳过")
            continue

        try:
            segment_data = json.loads(vision_response)
            if isinstance(segment_data, dict):
                if 'Start_Timestamp' not in segment_data:
                    segment_data['Start_Timestamp'] = f"{start_time:.1f}s"
                if 'End_Timestamp' not in segment_data:
                    segment_data['End_Timestamp'] = f"{end_time:.1f}s"
                all_segments.append(segment_data)
            else:
                detailed_logger.warning(f"间隔 {segment_id} 返回了无效的JSON格式")
        except json.JSONDecodeError as e:
            detailed_logger.error(f"解析间隔 {segment_id} 的JSON响应失败: {str(e)}")
            detailed_logger.error(f"原始响应: {vision_response[:200]}...")
            continue

        for frame_path in packet:
            if os.path.exists(frame_path):
                os.remove(frame_path)

        progress_bar.update(1)

    progress_bar.close()
    video_clip.close()

    result_filename = f'actionSummary_drivelm_{video_filename.split(".")[0]}.json'
    result_path = os.path.join(output_dir, result_filename)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 分析完成，结果保存到: {result_path}")
    return result_path

def transcribe_audio(audio_path, api_key, deployment, endpoint):
    try:
        url = f"{endpoint}/openai/deployments/{deployment}/audio/transcriptions?api-version=2024-02-01"
        headers = {
            "api-key": api_key,
        }
        
        with open(audio_path, 'rb') as audio_file:
            files = {
                'file': audio_file,
            }
            data = {
                'model': 'whisper-1',
                'response_format': 'text'
            }
            
            response = requests.post(url, headers=headers, files=files, data=data)
            
            if response.status_code == 200:
                return response.text.strip()
            else:
                detailed_logger.error(f"音频转录失败: {response.status_code} - {response.text}")
                return ""
                
    except Exception as e:
        detailed_logger.error(f"音频转录异常: {str(e)}")
        return ""

def retry_if_connection_error(exception):
    is_connection_error = isinstance(exception, (requests.exceptions.ConnectionError,
                                                 requests.exceptions.Timeout))
    if is_connection_error:
        detailed_logger.warning(f"API连接错误，将重试: {str(exception)}")
    return is_connection_error

# 🔧 平衡版GPT-4.1 vision analysis function
@retry(stop_max_attempt_number=2, wait_exponential_multiplier=2000, wait_exponential_max=60000,
       retry_on_exception=retry_if_connection_error)
def gpt41_vision_analysis_balanced(image_path, api_key, summary, trans, video_id, segment_id=None, speed_mode=False, start_time=0, end_time=10):
    detailed_logger.info(f"开始平衡版GPT-4.1视觉分析, 视频ID: {video_id}, 段落ID: {segment_id}, 图像数量: {len(image_path)}")
    
    encoded_images = []
    for path in image_path:
        with open(path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_images.append(encoded_string)
    
    frame_interval = 10
    frames_per_interval = len(image_path)
    
    if vision_api_type == "Azure":
        detailed_logger.info("使用平衡版Azure OpenAI GPT-4.1进行视觉分析")
        segment_id_str = segment_id if segment_id else f"Segment_{0:03d}"
        
        # 🔧 DriveLM风格Graph VQA PROMPT - 基于Graph Visual Question Answering方法论
        system_content = f"""You are DriveLM, an advanced Graph Visual Question Answering system for autonomous driving scene understanding.

**GRAPH VQA METHODOLOGY:**
You will analyze driving scenarios through structured graph reasoning, connecting perception, prediction, planning, and behavioral elements.

**GRAPH ELEMENTS TO IDENTIFY:**
1. **EGO VEHICLE**: The autonomous vehicle's current state and trajectory
2. **TRAFFIC PARTICIPANTS**: Vehicles, pedestrians, cyclists in the scene  
3. **ROAD INFRASTRUCTURE**: Lanes, traffic signs, intersections
4. **DYNAMIC RELATIONSHIPS**: Spatial and temporal relationships between elements
5. **RISK FACTORS**: Potential collision points and safety-critical events

**MULTI-STEP REASONING PROCESS:**

**STEP 1: Scene Graph Construction**
Build a structured representation of the current driving scenario identifying all key entities and their relationships.

**STEP 2: Temporal Analysis** 
Analyze the sequence of {frames_per_interval} frames to understand motion patterns and predict future states.

**STEP 3: Risk Assessment**
Evaluate potential safety-critical events, particularly focusing on sudden appearances (ghost probing).

**STEP 4: Graph-based Decision Making**
Use the constructed graph to make reasoned decisions about scene understanding.

**PRIMARY TASK: Ghost Probing Detection**
Ghost probing refers to the sudden appearance of pedestrians, vehicles, or objects that create immediate collision risk for the ego vehicle.

**DETECTION CRITERIA:**
- Sudden appearance within ego vehicle's trajectory
- Objects appearing from blind spots (behind parked cars, buildings, etc.)
- Rapid movement into the vehicle's path
- High collision risk scenarios requiring emergency response

**GRAPH-BASED ANALYSIS GUIDELINES:**
- **HIGH-CONFIDENCE Ghost Probing (use "ghost probing")**:
  * Node: Sudden object appearance within critical distance (<3m)
  * Edge: Direct collision trajectory with ego vehicle
  * Risk Level: CRITICAL requiring immediate emergency response

- **MODERATE Ghost Probing (use "potential ghost probing")**:
  * Node: Object appears at moderate distance (3-5m) 
  * Edge: Possible collision trajectory requiring attention
  * Risk Level: HIGH requiring significant response

- **NORMAL Traffic (descriptive terms)**:
  * Node: Expected traffic participants in predictable locations
  * Edge: Normal traffic flow relationships
  * Risk Level: LOW with standard driving responses

Your response should be a valid JSON object with the following EXACT structure (match this format precisely):
{{
    "video_id": "{video_id}",
    "segment_id": "{segment_id_str}",
    "Start_Timestamp": "{start_time:.1f}s",
    "End_Timestamp": "{end_time:.1f}s",
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
}}

Audio Transcription: {trans}
"""
        
        content = [{"type": "text", "text": system_content}]
        
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
        
        response = send_post_request(vision_endpoint, 
                                   vision_deployment, 
                                   openai_api_key, 
                                   data)
        return response
    
    else:  # OpenAI API
        detailed_logger.info("使用平衡版OpenAI GPT-4.1进行视觉分析")
        
        content = [
            {
                "type": "text", 
                "text": f"""You are VideoAnalyzerGPT. Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.

For ghost probing detection, consider TWO categories:

**1. HIGH-CONFIDENCE Ghost Probing (use "ghost probing")**:
- Object appears EXTREMELY close (within 1-2 vehicle lengths)
- Sudden appearance from blind spots in high-risk environments
- Requires IMMEDIATE emergency action

**2. POTENTIAL Ghost Probing (use "potential ghost probing")**:
- Object appears suddenly at moderate distance
- Unexpected movement requiring emergency braking
- Borderline cases where ghost probing is possible

**3. NORMAL Traffic (descriptive terms)**:
- Expected behaviors in intersections/crosswalks
- Normal lane changes and turns
- Predictable cyclist/pedestrian movement

Your response should be a valid JSON object with the following EXACT structure:
{{
    "video_id": "{video_id}",
    "segment_id": "{segment_id}",
    "Start_Timestamp": "{start_time:.1f}s",
    "End_Timestamp": "{end_time:.1f}s",
    "sentiment": "Positive/Negative/Neutral",
    "scene_theme": "Dramatic/Routine/Dangerous/Safe",
    "characters": "brief description of people in the scene",
    "summary": "comprehensive summary of the scene and what happens",
    "actions": "actions taken by the vehicle and driver responses",
    "key_objects": "numbered list: 1) Position: object description, distance, behavior impact 2) Position: object description, distance, behavior impact",
    "key_actions": "brief description of most important actions (use 'ghost probing', 'potential ghost probing', or descriptive terms)",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}

Audio Transcription: {trans}
"""
            }
        ]
        
        for encoded_image in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4.1",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 2000,
            "temperature": 0
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                               headers=headers, 
                               json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        else:
            detailed_logger.error(f"OpenAI API调用失败: {response.status_code} - {response.text}")
            return -1

def send_post_request(endpoint, deployment_name, api_key, data):
    if not endpoint.startswith('https://'):
        endpoint = f"https://{endpoint}.openai.azure.com"
    
    url = f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-15-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        else:
            detailed_logger.error(f"Azure API调用失败: {response.status_code} - {response.text}")
            return -1
    except Exception as e:
        detailed_logger.error(f"发送请求时发生异常: {str(e)}")
        return -1

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='平衡版GPT-4.1视频分析 - 保持高召回率')
    parser.add_argument('--single', type=str, help='处理单个视频文件')
    parser.add_argument('--interval', type=int, default=10, help='帧间隔（秒）')
    parser.add_argument('--frames', type=int, default=10, help='每间隔的帧数')
    parser.add_argument('--output-dir', type=str, default='result/gpt41-balanced', help='输出目录')
    
    args = parser.parse_args()
    
    if args.single:
        print(f"🧪 测试平衡版GPT-4.1 prompt")
        print(f"📁 视频: {args.single}")
        print(f"📂 输出目录: {args.output_dir}")
        print("=" * 50)
        
        result = AnalyzeVideo(args.single, args.interval, args.frames, False, args.output_dir)
        if result:
            print(f"✅ 平衡版处理成功: {result}")
        else:
            print("❌ 平衡版处理失败")

if __name__ == "__main__":
    main()