import re
from functools import partial  # 用于创建带参数的函数
import multiprocessing  # 用于多进程并行处理视频
import datetime  # 用于日志文件名
import traceback  # 用于获取详细错误信息
import tqdm  # 添加进度条库
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
    # 动态返回进程的帧目录
    return f'frames_process_{process_id}'

# 视频处理包装函数定义在模块顶层，以便多进程能正确访问
def process_video_wrapper(video_path, args):
    """
    视频处理的包装函数，用于多进程调用
    """
    try:
        video_name = os.path.basename(video_path)
        # 检查是否已经处理过该视频
        result_filename = f'actionSummary_{video_name.split(".")[0]}.json'
        result_path = os.path.join(args.output_dir, result_filename)
        
        if os.path.exists(result_path) and not args.no_skip and not args.retry_failed:
            print(f"视频 {video_name} 已经处理过，跳过")
            return (video_path, "skipped", None, 0)
        
        # 设置进程专用的帧目录
        global CURRENT_PROCESS_ID
        CURRENT_PROCESS_ID = multiprocessing.current_process().ident % 1000
        
        # 处理视频
        result = AnalyzeVideo(video_path, args.interval, args.frames, args.speed_mode, args.output_dir)
        
        # 清理进程专用的帧目录
        if args.separate_frame_dirs:
            frame_dir = get_process_frame_dir()
            if os.path.exists(frame_dir):
                import shutil
                shutil.rmtree(frame_dir)
        
        return (video_path, "success", result, 0)
        
    except Exception as e:
        error_msg = f"处理视频 {video_path} 时发生错误: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return (video_path, "error", error_msg, 1)

# 初始化日志记录器
def initialize_logger(log_level='INFO'):
    """
    初始化日志记录器
    """
    logger = logging.getLogger('video_processor')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
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
    @wraps(func)  # Preserves the name and docstring of the decorated function
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
        self.thread.daemon = True  # 设置为守护线程，确保主程序退出时不会被阻塞
        self.thread.start()
        if self.logger:
            self.logger.debug(f"Spinner thread started for: {self.message}")

    def stop(self):
        self.stop_spinner = True
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)  # 设置超时避免无限等待
            sys.stdout.write(
                '\r' + ' '*(len(self.message)+2) + '\r')  # 擦除spinner
            sys.stdout.flush()
            if self.logger:
                self.logger.debug(f"Spinner stopped: {self.message}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error stopping spinner: {str(e)}")
            # 即使出错也要尝试清除spinner显示
            try:
                sys.stdout.write('\r' + ' '*(len(self.message)+2) + '\r')
                sys.stdout.flush()
            except:
                pass

chapter_summary = {}
miss_arr = []

def extract_video_id(filename, logger=None):
    """
    从文件名中提取视频ID，适用于多种格式的文件名
    """
    if logger:
        logger.debug(f"提取视频ID，文件名: {filename}")

    try:
        # 验证输入
        if not filename or not isinstance(filename, str):
            error_msg = f"无效的文件名: {filename}"
            if logger:
                logger.error(error_msg)
            return f"unknown_{int(time.time())}"

        # 匹配DADA-2000-videos中的"images_XX_YYY.avi"格式
        dada_match = re.match(r"images_(\d+)_(\d+)\.avi", filename)
        if dada_match:
            video_id = f"dada_{dada_match.group(1)}_{dada_match.group(2)}"
            if logger:
                logger.debug(f"匹配DADA格式成功: {video_id}")
            return video_id

        # 匹配数字前缀格式的文件名，如"001_视频.mp4"
        num_prefix_match = re.match(r"(\d+)_", filename)
        if num_prefix_match:
            video_id = f"vid_{num_prefix_match.group(1)}"
            if logger:
                logger.debug(f"匹配数字前缀格式成功: {video_id}")
            return video_id

        # 去掉扩展名作为通用备选方案
        base_name = os.path.splitext(filename)[0]
        # 移除非法字符，以便作为ID使用
        video_id = re.sub(r'[^\w-]', '_', base_name)
        if logger:
            logger.debug(f"使用通用备选方案: {video_id}")
        return video_id
    except Exception as e:
        # 记录错误但继续执行，提供一个安全的默认值
        error_msg = f"提取视频ID时出错: {str(e)}, 回退到安全ID"
        if logger:
            logger.error(error_msg)
            logger.error(traceback.format_exc())

        # 生成一个唯一的ID作为fallback
        safe_id = f"unknown_{int(time.time())}_{hash(filename) % 10000 if filename else 0}"
        return safe_id

@log_execution_time
def AnalyzeVideo(vp, fi, fpi, speed_mode=False, output_dir='result/gpt-4.1'):
    video_filename = os.path.basename(vp)
    video_id = extract_video_id(video_filename)
    print(f"处理视频: {video_filename}, 视频ID: {video_id}")

    # 程序运行在多进程模式下，使用进程特定的帧目录
    global CURRENT_PROCESS_ID
    if multiprocessing.current_process().name != 'MainProcess':
        # 在多进程模式下，使用进程特定的帧目录
        frames_dir = get_process_frame_dir()
    else:
        # 在主进程或单进程模式下，使用默认目录
        frames_dir = 'frames'

    # 确保帧目录存在
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 从视频中提取帧
    video_clip = VideoFileClip(vp)
    duration = video_clip.duration
    detailed_logger.info(f"视频时长: {duration:.2f}秒")

    # 计算总间隔数
    total_intervals = int(duration / fi)
    if duration % fi > 0:
        total_intervals += 1

    detailed_logger.info(f"将创建 {total_intervals} 个间隔，每间隔 {fi} 秒，每间隔 {fpi} 帧")

    # 添加进度条
    progress_bar = tqdm.tqdm(total=total_intervals, desc="处理视频间隔", unit="interval")

    # 存储所有间隔的分析结果
    all_segments = []

    # 处理每个间隔
    for interval_index in range(total_intervals):
        segment_id = f"segment_{interval_index:03d}"
        start_time = interval_index * fi
        end_time = min(start_time + fi, duration)
        
        detailed_logger.info(f"处理间隔 {interval_index + 1}/{total_intervals}: {start_time:.1f}s - {end_time:.1f}s")

        # 为当前间隔提取帧
        packet = []
        for frame_index in range(fpi):
            frame_time = start_time + (frame_index * (end_time - start_time) / fpi)
            if frame_time >= duration:
                break
            
            frame_filename = f"{frames_dir}/frame_at_{frame_time:.1f}s.jpg"
            
            # 提取帧
            frame = video_clip.get_frame(frame_time)
            cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            packet.append(frame_filename)
            detailed_logger.debug(f"提取帧: {frame_filename}")

        if not packet:
            detailed_logger.warning(f"间隔 {segment_id} 没有提取到帧，跳过")
            continue

        # 提取音频转录
        current_transcription = ""
        if video_clip.audio is not None:
            spinner = Spinner(f"正在转录间隔 {segment_id} 的音频...")
            spinner.start()
            
            try:
                audio_filename = f"{frames_dir}/audio_segment_{segment_id}.wav"
                audio_clip = video_clip.subclip(start_time, end_time)
                audio_clip.audio.write_audiofile(audio_filename, verbose=False, logger=None)
                
                # 转录音频
                tscribe = transcribe_audio(audio_filename, azure_whisper_key, azure_whisper_deployment, azure_whisper_endpoint)
                current_transcription = tscribe
                
                # 清理音频文件
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
                    
            except Exception as e:
                detailed_logger.error(f"音频转录失败: {str(e)}")
                current_transcription = ""
            finally:
                spinner.stop()
        else:
            print("视频没有音轨，跳过音频提取和转录")

        # 使用GPT-4.1 Vision分析当前间隔的帧
        spinner = Spinner(f"正在分析间隔 {segment_id} 的 {len(packet)} 帧图像...")
        spinner.start()

        # 调用GPT-4.1 Vision进行分析
        vision_response = gpt41_vision_analysis(
            packet, openai_api_key, "", current_transcription, video_id, segment_id, speed_mode, start_time, end_time)

        spinner.stop()

        if vision_response == -1:
            print(f"警告: 间隔 {segment_id} 的视觉分析失败，跳过")
            continue

        # 解析JSON响应
        try:
            segment_data = json.loads(vision_response)
            if isinstance(segment_data, dict):
                # 确保时间戳已经包含在响应中，如果没有则添加
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

        # 清理当前间隔的帧文件
        for frame_path in packet:
            if os.path.exists(frame_path):
                os.remove(frame_path)

        # 更新进度条
        progress_bar.update(1)

    # 关闭进度条
    progress_bar.close()

    # 清理视频剪辑对象
    video_clip.close()

    # 保存结果
    result_filename = f'actionSummary_{video_filename.split(".")[0]}.json'
    result_path = os.path.join(output_dir, result_filename)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 分析完成，结果保存到: {result_path}")
    return result_path

def transcribe_audio(audio_path, api_key, deployment, endpoint):
    """转录音频文件"""
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
    """判断是否为连接错误，需要重试"""
    is_connection_error = isinstance(exception, (requests.exceptions.ConnectionError,
                                                 requests.exceptions.Timeout))
    if is_connection_error:
        detailed_logger.warning(f"API连接错误，将重试: {str(exception)}")
    return is_connection_error

# GPT-4.1 vision analysis function
@retry(stop_max_attempt_number=2, wait_exponential_multiplier=2000, wait_exponential_max=60000,
       retry_on_exception=retry_if_connection_error)
def gpt41_vision_analysis(image_path, api_key, summary, trans, video_id, segment_id=None, speed_mode=False, start_time=0, end_time=10):
    detailed_logger.info(f"开始GPT-4.1视觉分析, 视频ID: {video_id}, 段落ID: {segment_id}, 图像数量: {len(image_path)}")
    
    # 编码图像
    encoded_images = []
    for path in image_path:
        with open(path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_images.append(encoded_string)
    
    # 设置参数
    frame_interval = 10  # 假设为10秒间隔
    frames_per_interval = len(image_path)
    
    if vision_api_type == "Azure":
        detailed_logger.info("使用Azure OpenAI GPT-4.1进行视觉分析")
        segment_id_str = segment_id if segment_id else f"Segment_{0:03d}"
        
        system_content = f"""You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the video.

IMPORTANT: For ghost probing detection, look for:
- Objects (people, vehicles, animals) that suddenly appear from blind spots
- Sudden movements crossing the vehicle's path
- Unexpected intrusions into the driving lane
- Objects emerging from concealed positions (behind parked cars, structures)
- Any scenario where an object "probes" or tests the vehicle's reaction

If you detect any ghost probing behavior, explicitly mention "ghost probing" in your key_actions field.

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
    "key_actions": "brief description of most important actions (use 'ghost probing' if detected)",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}

Audio Transcription: {trans}
"""
        
        # 构建消息内容
        content = [{"type": "text", "text": system_content}]
        
        # 添加图像
        for encoded_image in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })
        
        # 构建请求数据
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        # 发送请求
        response = send_post_request(vision_endpoint, 
                                   vision_deployment, 
                                   openai_api_key, 
                                   data)
        return response
    
    else:  # OpenAI API
        detailed_logger.info("使用OpenAI GPT-4.1进行视觉分析")
        
        # 构建消息内容
        content = [
            {
                "type": "text", 
                "text": f"""You are VideoAnalyzerGPT. Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are then to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the video.

IMPORTANT: For ghost probing detection, look for:
- Objects that suddenly appear from blind spots
- Sudden movements crossing the vehicle's path  
- Unexpected intrusions into the driving lane
- Objects emerging from concealed positions
- Any scenario where an object "probes" or tests the vehicle's reaction

If you detect any ghost probing behavior, explicitly mention "ghost probing" in your key_actions field.

Your response should be a valid JSON object with the following EXACT structure (match this format precisely):
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
    "key_actions": "brief description of most important actions (use 'ghost probing' if detected)",
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
        
        # 添加图像
        for encoded_image in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })
        
        # 构建请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4.1",  # 使用GPT-4.1模型
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.3
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
    """发送POST请求到Azure OpenAI"""
    # 确保endpoint格式正确
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

def process_all_videos(video_folder, frame_interval=10, frames_per_interval=10, limit=None, skip_existing=True, 
                       start_at=0, retry_failed=False, progress_save_interval=1, output_dir='result/gpt-4.1'):
    """批量处理指定文件夹中的所有视频文件"""
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取视频文件列表
    video_files = []
    for filename in os.listdir(video_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(os.path.join(video_folder, filename))
    
    # 排序确保处理顺序一致
    video_files.sort()
    
    # 应用限制和起始位置
    if start_at > 0:
        video_files = video_files[start_at:]
    
    if limit:
        video_files = video_files[:limit]
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理视频
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for i, video_path in enumerate(video_files):
        try:
            print(f"\n[{i+1}/{len(video_files)}] 处理视频: {os.path.basename(video_path)}")
            
            # 检查是否已经处理过
            video_name = os.path.basename(video_path)
            result_filename = f'actionSummary_{video_name.split(".")[0]}.json'
            result_path = os.path.join(output_dir, result_filename)
            
            if os.path.exists(result_path) and skip_existing and not retry_failed:
                print("已处理过，跳过")
                skipped_count += 1
                continue
            
            # 处理视频
            result = AnalyzeVideo(video_path, frame_interval, frames_per_interval, False, output_dir)
            
            if result:
                success_count += 1
                print(f"✅ 成功处理")
            else:
                error_count += 1
                print(f"❌ 处理失败")
                
        except Exception as e:
            error_count += 1
            print(f"❌ 处理视频 {video_path} 时发生异常: {str(e)}")
            traceback.print_exc()
    
    print(f"\n处理完成:")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"跳过: {skipped_count}")
    
    return success_count, error_count, skipped_count

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='批量处理视频文件进行GPT-4.1分析')
    parser.add_argument('--folder', type=str, default='DADA-2000-videos', help='视频文件夹路径')
    parser.add_argument('--interval', type=int, default=10, help='帧间隔（秒）')
    parser.add_argument('--frames', type=int, default=10, help='每间隔的帧数')
    parser.add_argument('--single', type=str, help='处理单个视频文件')
    parser.add_argument('--limit', type=int, help='限制处理的视频数量')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已处理的视频')
    parser.add_argument('--start-at', type=int, default=0, help='从第N个视频开始处理')
    parser.add_argument('--retry-failed', action='store_true', help='重试之前失败的视频')
    parser.add_argument('--output-dir', type=str, default='result/gpt-4.1', help='输出目录')
    
    args = parser.parse_args()
    
    if args.single:
        # 处理单个视频
        result = AnalyzeVideo(args.single, args.interval, args.frames, False, args.output_dir)
        if result:
            print(f"✅ 单个视频处理成功: {result}")
        else:
            print("❌ 单个视频处理失败")
    else:
        # 批量处理
        success, error, skipped = process_all_videos(
            args.folder, 
            args.interval, 
            args.frames, 
            args.limit, 
            not args.no_skip, 
            args.start_at, 
            args.retry_failed, 
            1, 
            args.output_dir
        )
        
        print(f"\n最终统计:")
        print(f"成功: {success}")
        print(f"失败: {error}")
        print(f"跳过: {skipped}")

if __name__ == "__main__":
    main()