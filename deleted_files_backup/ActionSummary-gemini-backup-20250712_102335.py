#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 基于ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper-batch.py的Gemini 1.5 Flash版本
# 将原始GPT-4 Vision API替换为Gemini 1.5 Flash API

import cv2
import os
import base64
import requests
from moviepy.editor import VideoFileClip
import logging
import json
import sys
import threading
from retrying import retry
logging.getLogger('moviepy').setLevel(logging.ERROR)
import time
from functools import wraps
from dotenv import load_dotenv
import time
import video_utilities as vu
from jinja2 import Environment, FileSystemLoader
import numpy as np
import tqdm  # 添加进度条库
import traceback  # 用于获取详细错误信息
import datetime  # 用于日志文件名
import multiprocessing  # 用于多进程并行处理视频
from functools import partial  # 用于创建带参数的函数
import re

# 添加Gemini所需的库
import google.generativeai as genai
from google.genai import types

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
        
        start_time = time.time()
        # 处理视频
        result = AnalyzeVideo(video_path, args.interval, args.frames, args.speed_mode)
        processing_time = time.time() - start_time
        
        return (video_path, "success", result, processing_time)
    except Exception as e:
        error_msg = f"处理视频失败: {str(e)}"
        print(error_msg)
        if 'detailed_logger' in globals():
            detailed_logger.error(error_msg)
            detailed_logger.error(traceback.format_exc())
        return (video_path, "failed", str(e), 0)

final_arr=[]
load_dotenv(dotenv_path=".env", override=True)  # 加 override=True，确保更新

# 配置详细日志记录器
def setup_logger(name, log_file, level=logging.INFO, append=True, console_output=True, format_str=None):
    """设置一个日志记录器，将信息记录到指定文件
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        append: 是否追加模式，否则覆盖
        console_output: 是否同时输出到控制台
        format_str: 自定义日志格式，如果为None则使用默认格式
    
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的处理器，避免重复
    if logger.handlers:
        logger.handlers = []
    
    # 创建文件处理器
    mode = 'a' if append else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode, encoding='utf-8')
    
    # 创建格式化器并添加到处理器
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)
    file_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    
    # 添加控制台处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# 创建日志记录器
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# 以日期时间命名日志文件，确保唯一性
log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
detailed_log_file = os.path.join(logs_dir, f'detailed_processing_{log_timestamp}.log')
# 创建详细日志记录器
detailed_logger = setup_logger('detailed_logger', detailed_log_file, level=logging.DEBUG)
detailed_logger.info(f"开始处理视频批处理任务，日志时间戳: {log_timestamp}")

# 从环境变量获取Gemini API设置
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")  # 默认使用1.5-flash模型

# 从环境变量获取其他设置（依然保留以兼容现有代码）
# Azure speech key
speech_key = os.environ.get("AZURE_SPEECH_KEY")

# Azure whisper key
AZ_WHISPER = os.environ.get("AZURE_WHISPER_KEY")

# Azure whisper deployment name
azure_whisper_deployment = os.environ.get("AZURE_WHISPER_DEPLOYMENT")

# Azure whisper endpoint (just name)
azure_whisper_endpoint = os.environ.get("AZURE_WHISPER_ENDPOINT")

# Audio API type (OpenAI, Azure)
audio_api_type = os.environ.get("AUDIO_API_TYPE")

# 设置Gemini API客户端
if not GEMINI_API_KEY:
    detailed_logger.error("GEMINI_API_KEY未设置，请在.env文件中添加")
    raise ValueError("GEMINI_API_KEY未设置，请在.env文件中添加GEMINI_API_KEY=你的密钥")

# 配置Gemini客户端
detailed_logger.info(f"初始化Gemini客户端，使用模型: {GEMINI_MODEL}")
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    detailed_logger.info("Gemini客户端初始化成功")
except Exception as e:
    detailed_logger.error(f"Gemini客户端初始化失败: {str(e)}")
    raise ValueError(f"Gemini客户端初始化失败: {str(e)}")

def log_execution_time(func):
    @wraps(func)  # Preserves the name and docstring of the decorated function
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to complete.")
        detailed_logger.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to complete.")
        return result
    return wrapper

class Spinner:
    def __init__(self, message="Processing...", logger=None):
        self.spinner_symbols = "|/-\\"
        self.idx = 0
        self.message = message
        self.stop_spinner = False
        self.logger = logger
        if self.logger:
            self.logger.debug(f"Spinner started: {message}")

    def spinner_task(self):
        try:
            while not self.stop_spinner:
                sys.stdout.write(f"\r{self.message} {self.spinner_symbols[self.idx % len(self.spinner_symbols)]}")
                sys.stdout.flush()
                time.sleep(0.1)
                self.idx += 1
        except Exception as e:
            if self.logger:
                self.logger.error(f"Spinner task error: {str(e)}")

    def start(self):
        self.stop_spinner = False
        self.thread = threading.Thread(target=self.spinner_task)
        self.thread.daemon = True  # 设置为守护线程，确保主程序退出时不会被阻塞
        self.thread.start()
        if self.logger:
            self.logger.debug(f"Spinner thread started for: {self.message}")

    def stop(self):
        self.stop_spinner = True
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)  # 设置超时避免无限等待
            sys.stdout.write('\r' + ' '*(len(self.message)+2) + '\r')  # 擦除spinner
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
    :param filename: 文件名（如 `001_video.mp4` 或 `images_11_085.avi`）
    :param logger: 日志记录器
    :return: 提取的视频ID
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
def AnalyzeVideo(vp, fi, fpi, speed_mode=False):  #fpi is frames per interval, fi is frame interval
    video_filename = os.path.basename(vp)
    video_id = extract_video_id(video_filename)
    print(f"处理视频: {video_filename}, 视频ID: {video_id}")

    # 如果启用了速度模式，减少帧数
    if speed_mode:
        # 速度模式下减少50%的帧数，但至少保留3帧
        fpi = max(3, int(fpi * 0.5))
        print(f"速度模式已启用: 每个间隔仅处理 {fpi} 帧")
        detailed_logger.info(f"速度模式已启用: 每个间隔仅处理 {fpi} 帧")

    segment_counter = 0
    # Constants
    video_path = vp  # Replace with your video path
    output_frame_dir = 'frames'
    output_audio_dir = 'audio'
    global_transcript=""
    transcriptions_dir = 'transcriptions'
    frame_interval = fi  # seconds 180
    frames_per_interval = fpi
    totalData=""
    # 用于存储所有间隔的分析结果
    all_intervals_results = []
    
    # Ensure output directories exist
    for directory in [output_frame_dir, output_audio_dir, transcriptions_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Encode image to base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # 修改后的函数处理Gemini模型的响应
    def create_vision_analysis(image_paths, summary, trans, video_id, segment_id=None):
        """
        调用Gemini视觉分析API并处理响应
        
        Args:
            image_paths: 图像路径列表
            summary: 摘要文本
            trans: 音频转录文本
            video_id: 视频ID
            segment_id: 段落ID
            
        Returns:
            处理后的JSON字符串或错误消息
        """
        detailed_logger.info(f"开始创建视觉分析, 视频ID: {video_id}, 段落ID: {segment_id}")
        
        # 使用 gemini_vision_analysis 函数并处理其返回值
        response = gemini_vision_analysis(image_paths, summary, trans, video_id, segment_id)
        
        # 检查是否返回了错误字典
        if isinstance(response, dict) and 'error' in response:
            error_msg = f"API调用返回错误: {response.get('error')}"
            detailed_logger.error(error_msg)
            # 创建一个包含错误信息的JSON对象
            error_result = {
                "video_id": video_id,
                "segment_id": segment_id if segment_id else f"unknown_segment",
                "error": response.get('error'),
                "status": "error",
                "timestamp": time.time()
            }
            return json.dumps(error_result, ensure_ascii=False)
        
        # 检查响应类型
        detailed_logger.debug(f"API响应类型: {type(response)}")
        
        try:
            # 尝试解析响应的JSON内容
            if hasattr(response, 'text'):
                # 如果返回的是Gemini对象，直接获取文本内容
                content = response.text
                detailed_logger.info("成功从Gemini响应中提取内容")
                
                # 尝试将内容解析为JSON
                try:
                    # 检查内容是否为JSON格式
                    if content.strip().startswith('{') and content.strip().endswith('}'):
                        json_content = json.loads(content)
                        # 确保包含视频ID和段落ID
                        if 'video_id' not in json_content:
                            json_content['video_id'] = video_id
                        if 'segment_id' not in json_content and segment_id:
                            json_content['segment_id'] = segment_id
                        return json.dumps(json_content, ensure_ascii=False)
                    else:
                        # 可能包含代码块或其他格式
                        detailed_logger.debug("内容不是直接的JSON格式，尝试提取")
                        # 尝试查找并提取JSON部分
                        json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
                        json_match = re.search(json_pattern, content)
                        if json_match:
                            json_content = json.loads(json_match.group(1))
                            # 确保包含视频ID和段落ID
                            if 'video_id' not in json_content:
                                json_content['video_id'] = video_id
                            if 'segment_id' not in json_content and segment_id:
                                json_content['segment_id'] = segment_id
                            return json.dumps(json_content, ensure_ascii=False)
                        else:
                            # 找不到JSON，返回原始内容
                            detailed_logger.warning("无法从内容中提取JSON，返回原始内容")
                            return content
                except json.JSONDecodeError as je:
                    detailed_logger.warning(f"内容不是有效的JSON: {str(je)}")
                    return content
            elif isinstance(response, dict):
                # 已经是字典格式
                # 确保包含视频ID和段落ID
                if 'video_id' not in response:
                    response['video_id'] = video_id
                if 'segment_id' not in response and segment_id:
                    response['segment_id'] = segment_id
                return json.dumps(response, ensure_ascii=False)
            else:
                # 其他响应类型，返回字符串形式
                return str(response)
        except Exception as e:
            error_msg = f"提取分析结果时出错: {str(e)}"
            detailed_logger.error(error_msg)
            detailed_logger.error(traceback.format_exc())
            
            # 保存原始响应以便调试
            debug_file = os.path.join(logs_dir, f'response_extract_error_{video_id}_{segment_id}.txt')
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(str(response))
            detailed_logger.info(f"已保存出错的原始响应到 {debug_file}")
            
            # 返回一个包含错误信息的JSON对象
            error_result = {
                "video_id": video_id,
                "segment_id": segment_id if segment_id else f"unknown_segment",
                "error": f"API响应解析失败: {str(e)}",
                "status": "error",
                "timestamp": time.time()
            }
            return json.dumps(error_result, ensure_ascii=False)

    # 重试策略: 5次重试，每次等待时间翻倍，从2秒开始
    # 只有在连接错误或超时时才重试
    def retry_if_connection_error(exception):
        is_connection_error = isinstance(exception, (requests.exceptions.ConnectionError, 
                                                   requests.exceptions.Timeout))
        if is_connection_error:
            detailed_logger.warning(f"API连接错误，将重试: {str(exception)}")
        return is_connection_error
    
    # Gemini视觉分析函数
    @retry(stop_max_attempt_number=5, wait_exponential_multiplier=2000, wait_exponential_max=60000, 
           retry_on_exception=retry_if_connection_error)
    def gemini_vision_analysis(image_paths, summary, trans, video_id, segment_id=None, speed_mode=False):
        detailed_logger.info(f"开始Gemini视觉分析, 视频ID: {video_id}, 段落ID: {segment_id}, 图像数量: {len(image_paths)}")
        
        # 检查关键参数
        if not image_paths or len(image_paths) == 0:
            detailed_logger.error("输入图像路径为空，无法进行分析")
            return {"error": "No images provided for analysis", "status": "error"}
        
        if not GEMINI_API_KEY:
            detailed_logger.error("Gemini API密钥为空，无法进行API调用")
            return {"error": "Gemini API key is missing", "status": "error"}
        
        # 如果启用了速度模式，则减少图像数量
        if speed_mode and len(image_paths) > 3:
            # 保留首、中、尾的三张关键图像
            first_img = image_paths[0]
            middle_idx = len(image_paths) // 2
            last_img = image_paths[-1]
            image_paths = [first_img, image_paths[middle_idx], last_img]
            detailed_logger.info(f"速度模式: 减少图像数量到3张关键帧")
        
        # 创建Gemini模型的提示内容
        prompt_parts = []
        
        # 添加音频转录信息
        prompt_parts.append(f"Audio Transcription for last {frame_interval} seconds: {trans}")
        prompt_parts.append(f"Next are the {len(image_paths)} frames from the last {frame_interval} seconds of the video:")
        
        # 加载图像
        valid_images = 0
        invalid_images = 0
        image_objects = []
        
        for img_path in image_paths:
            try:
                if not os.path.exists(img_path):
                    detailed_logger.warning(f"图像文件不存在: {img_path}")
                    invalid_images += 1
                    continue
                
                # 使用Gemini API的方式加载图像
                with open(img_path, 'rb') as f:
                    image_bytes = f.read()
                
                # 添加图像帧描述
                prompt_parts.append(f"Below this is {os.path.basename(img_path)} (s is for seconds). use this to provide timestamps and understand time")
                
                # 创建图像对象并添加到列表
                image_obj = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
                image_objects.append(image_obj)
                valid_images += 1
                
            except Exception as e:
                error_msg = f"处理图像时出错 {img_path}: {str(e)}"
                detailed_logger.error(error_msg)
                detailed_logger.error(traceback.format_exc())
                invalid_images += 1
                continue
        
        if valid_images == 0:
            detailed_logger.error(f"所有图像处理失败，无法进行分析。总共尝试了 {len(image_paths)} 个图像。")
            return {"error": "All images failed to process", "status": "error", "invalid_count": invalid_images}
        else:
            detailed_logger.info(f"成功处理 {valid_images}/{len(image_paths)} 个图像，准备进行分析")
        
        # 构建最终的提示内容列表
        contents = []
        
        # 系统指令：替换原有的GPT-4 vision系统提示
        segment_id_str = segment_id if segment_id else f"segment_{segment_counter:03d}"
        
        system_prompt = f"""You are VideoAnalyzerGPT analyzing a series of SEQUENCIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the portion of the video you are considering.

Direction - Please identify the objects in the image based on their position in the image itself. Do not assume your own position within the image. Treat the left side of the image as 'left' and the right side of the image as 'right'. Assume the viewpoint is standing from at the bottom center of the image. Describe whether the objects are on the left or right side of this central point, left is left, and right is right. For example, if there is a car on the left side of the image, state that the car is on the left.

**Task 1: Identify and Predict potential "Ghost Probing",Cut-in,left/right-side overtaken etc Behavior**

Note: Ghost Probing behavior refers to a person suddenly darting out from behind an obstacle, creating a dangerous situation with little reaction time. Cut-in happens when a vehicle from an adjacent lane merges abruptly. Left/right-side overtaken refers to another vehicle passing the ego vehicle safely.

**Task 2: Explain Current Driving Actions**
Analyze the current actions in the video frames, detailing why the vehicle is moving at a certain speed or direction.

**Task 3: Predict Next Driving Action**
Based on what you see, predict the most likely next actions in terms of speed control and lane control.

**Task 4: Ensure Consistency Between Key Objects and Key Actions**
When labeling a key action (like ghost probing), make sure to include the relevant objects causing this action.

Always return a single JSON object with the following fields:
- video_id: "{video_id}"
- segment_id: "{segment_id_str}"
- Start_Timestamp and End_Timestamp: derived from frame names
- summary: detailed description of what's happening
- actions: explanation of current vehicle actions
- key_objects: list of important objects affecting the vehicle
- key_actions: danger classification (ghost probing, cut-in, left/right-side overtaken, or none)
- next_action: JSON object with speed_control, direction_control, and lane_control fields

All text must be in English. Return only valid JSON."""

        # 替换占位符
        system_prompt = system_prompt.replace("{video_id}", video_id)
        system_prompt = system_prompt.replace("{segment_id_str}", segment_id_str)
                
        # 设置初始的系统提示
        # 注意: Gemini 1.5 Flash API使用safety settings来控制系统行为
        safety_settings = [
            {
                "category": "harassment",
                "threshold": "block_only_high"
            },
            {
                "category": "hate_speech",
                "threshold": "block_only_high"
            },
            {
                "category": "dangerous_content",
                "threshold": "block_only_high"
            }
        ]
        
        # 重试机制
        max_retries = 3
        retry_count = 0
        retry_delay = 5  # 初始延迟5秒
        last_error = None
        
        while retry_count < max_retries:
            try:
                detailed_logger.info(f"尝试Gemini API调用，尝试次数: {retry_count + 1}/{max_retries}")
                
                # 创建模型实例，使用 system instructions
                model = gemini_client.models.get_model(GEMINI_MODEL)
                
                # 准备请求内容 - 首先添加文本提示
                contents = [{"text": p} for p in prompt_parts]
                
                # 添加图像到内容
                for img_obj in image_objects:
                    contents.append(img_obj)
                
                # 发送Gemini API请求
                generation_config = {
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 4096
                }
                
                response = model.generate_content(
                    contents=contents,
                    system_instruction=system_prompt,
                    safety_settings=safety_settings,
                    generation_config=generation_config
                )
                
                # 检查响应是否成功
                if hasattr(response, 'text'):
                    detailed_logger.info("Gemini API调用成功")
                    return response
                else:
                    detailed_logger.warning(f"API响应格式不符合预期: {response}")
                    # 创建一个基本的错误对象并返回
                    return {"error": "API response format unexpected", "status": "error"}
                    
            except Exception as e:
                detailed_logger.error(f"Gemini API请求异常: {str(e)}")
                detailed_logger.error(traceback.format_exc())
                last_error = str(e)
                
                # 特定错误类型处理
                if "rate limit" in str(e).lower():
                    detailed_logger.warning(f"API速率限制错误，等待{retry_delay}秒后重试")
                    time.sleep(retry_delay * 2)  # 速率限制错误等待更长时间
                elif "internal" in str(e).lower():
                    detailed_logger.warning(f"API服务器错误，等待{retry_delay}秒后重试")
                    time.sleep(retry_delay)
                else:
                    time.sleep(retry_delay)
                
                retry_delay *= 2  # 指数退避
                retry_count += 1
        
        # 所有重试尝试都失败
        error_msg = f"在{max_retries}次尝试后，Gemini API调用仍然失败: {last_error}"
        detailed_logger.error(error_msg)
        return {"error": error_msg, "status": "error", "retries_exhausted": True} 

    def update_chapter_summary(new_json_string):
        global chapter_summary
        if new_json_string.startswith('json'):
        # Remove the first occurrence of 'json' from the response text
            new_json_string = new_json_string[4:]
        else:
            new_json_string = new_json_string
        # Assuming new_json_string is the JSON format string returned from your API call
        new_chapters_list = json.loads(new_json_string)

        # Iterate over the list of new chapters
        for chapter in new_chapters_list:
            chapter_title = chapter['title']
            # Update the chapter_summary with the new chapter
            chapter_summary[chapter_title] = chapter

        # Get keys of the last three chapters
        last_three_keys = list(chapter_summary.keys())[-3:]
        # Get the last three chapters as an array
        last_three_chapters = [chapter_summary[key] for key in last_three_keys]

        return last_three_chapters

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    print(f"视频帧率: {fps}，总帧数: {total_frames}")

    # 加载视频音频
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration  # Duration of the video in seconds
    print(f"视频时长: {video_duration} 秒")
    
    # 计算需要处理的间隔数量
    num_intervals = max(1, int(np.ceil(video_duration / frame_interval)))
    print(f"需要处理的间隔数量: {num_intervals} 个，每个间隔 {frame_interval} 秒")
    
    # 创建处理进度条
    progress_bar = tqdm.tqdm(total=num_intervals, desc="处理视频", unit="间隔")

    # 初始化处理状态
    current_interval = 0
    intervals_processed = 0
    global final_arr
    
    # 按间隔处理视频
    while current_interval < video_duration:
        # 计算当前间隔的开始和结束时间
        interval_start = current_interval
        interval_end = min(interval_start + frame_interval, video_duration)
        
        # 清空当前间隔的帧缓存
        packet = []
        
        # 计算当前间隔需要捕获的帧
        frames_to_capture = frames_per_interval
        
        # 为短于完整间隔的最后一段视频调整帧数
        if interval_end - interval_start < frame_interval:
            # 最后一段视频可能不足一个完整间隔，但我们仍然使用相同数量的帧
            # 这样可以保持分析质量
            print(f"最后一段视频长度为 {interval_end - interval_start:.2f} 秒，小于间隔设置 {frame_interval} 秒")
        
        # 计算当前间隔内的均匀采样点
        if frames_to_capture > 0:
            interval_duration = interval_end - interval_start
            capture_points = [interval_start + (i * interval_duration / frames_to_capture) for i in range(frames_to_capture)]
        else:
            capture_points = []
        
        # 显示当前处理的间隔信息
        segment_id = f"segment_{segment_counter:03d}"
        print(f"\n处理间隔 {segment_id}: {interval_start:.2f}s - {interval_end:.2f}s，采集 {len(capture_points)} 帧")
        
        # 捕获当前间隔的所有帧
        for capture_time in capture_points:
            # 计算对应的帧号
            frame_number = int(capture_time * fps)
            
            # 确保帧号在有效范围内
            if frame_number >= total_frames:
                frame_number = total_frames - 1
            
            # 定位到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                print(f"警告: 无法读取帧 {frame_number}，跳过")
                continue
            
            # 保存帧
            frame_name = f'frame_at_{capture_time:.1f}s.jpg'
            frame_path = os.path.join(output_frame_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            packet.append(frame_path)
            print(f"已捕获帧: {frame_path}")
        
        # 处理当前间隔的音频和图像
        if packet:  # 确保有捕获到帧
            current_transcription = ""
            
            # 提取并转录音频
            if video_clip.audio is not None:
                audio_name = f'audio_at_{interval_start:.1f}s.mp3'
                audio_path = os.path.join(output_audio_dir, audio_name)
                
                # 提取当前间隔的音频
                audio_clip = video_clip.subclip(interval_start, interval_end)
                audio_clip.audio.write_audiofile(audio_path, codec='mp3', verbose=False, logger=None)

                # 转录音频
                spinner = Spinner(f"正在转录间隔 {segment_id} 的音频...")
                spinner.start()
                
                @retry(stop_max_attempt_number=3)
                def transcribe_audio(audio_path, endpoint, api_key, deployment_name):
                    url = f"{endpoint}/openai/deployments/{deployment_name}/audio/transcriptions?api-version=2023-09-01-preview"

                    headers = {
                        "api-key": api_key,
                    }
                    json = {
                        "file": (audio_path.split("/")[-1], open(audio_path, "rb"), "audio/mp3"),
                    }
                    data = {
                        'response_format': (None, 'verbose_json')
                    }
                    response = requests.post(url, headers=headers, files=json, data=data)

                    return response

                if audio_api_type == "Azure":
                    response = transcribe_audio(audio_path, azure_whisper_endpoint, AZ_WHISPER, azure_whisper_deployment)
                else:
                    from openai import OpenAI
                    client = OpenAI()

                    audio_file = open(audio_path, "rb")
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                    )

                tscribe = ""
                # 处理转录响应
                if audio_api_type == "Azure":
                    try:
                        for item in response.json()["segments"]:
                            tscribe += str(round(item["start"], 2)) + "s - " + str(round(item["end"], 2)) + "s: " + item["text"] + "\n"
                    except:
                        tscribe += ""
                else:
                    for item in response.segments:
                        tscribe += str(round(item["start"], 2)) + "s - " + str(round(item["end"], 2)) + "s: " + item["text"] + "\n"
                
                global_transcript += "\n" + tscribe
                current_transcription = tscribe
                spinner.stop()
            else:
                print("视频没有音轨，跳过音频提取和转录")
            
            # 使用Gemini分析当前间隔的帧
            spinner = Spinner(f"正在分析间隔 {segment_id} 的 {len(packet)} 帧图像...")
            spinner.start()

            # 调用Gemini进行分析，传入segment_id参数
            vision_response = gemini_vision_analysis(packet, "", current_transcription, video_id, segment_id, speed_mode)
            
            if isinstance(vision_response, dict) and 'error' in vision_response:
                print(f"警告: 间隔 {segment_id} 的视觉分析失败: {vision_response.get('error')}，跳过")
                spinner.stop()
                segment_counter += 1
                current_interval = interval_end
                intervals_processed += 1
                progress_bar.update(1)
                continue
            
            # 处理分析结果
            try:
                # 直接获取Gemini响应的文本内容
                if hasattr(vision_response, 'text'):
                    vision_analysis = vision_response.text
                    detailed_logger.info("成功从Gemini响应中提取内容")
                else:
                    # 如果不是标准响应对象，尝试转换
                    vision_analysis = str(vision_response)
                    detailed_logger.warning(f"非标准响应对象: {type(vision_response)}")
                
                # 保存原始JSON以便调试
                debug_json_path = os.path.join(logs_dir, f'raw_json_{video_id}_{segment_id}.json')
                with open(debug_json_path, 'w', encoding='utf-8') as f:
                    f.write(str(vision_analysis))
                detailed_logger.debug(f"保存原始JSON到 {debug_json_path}")
                
                # 尝试解析JSON内容
                try:
                    # 检查内容是否为JSON格式或包含JSON
                    if vision_analysis.strip().startswith('{') and vision_analysis.strip().endswith('}'):
                        data = json.loads(vision_analysis.strip())
                        detailed_logger.info("成功解析JSON")
                    else:
                        # 尝试从文本中提取JSON
                        json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
                        json_match = re.search(json_pattern, vision_analysis)
                        if json_match:
                            data = json.loads(json_match.group(1))
                            detailed_logger.info("从Markdown代码块中提取JSON成功")
                        else:
                            # 最后尝试：使用简单的正则表达式查找JSON对象
                            json_match = re.search(r'\{.*\}', vision_analysis, re.DOTALL)
                            if json_match:
                                data = json.loads(json_match.group(0))
                                detailed_logger.info("使用正则表达式提取JSON成功")
                            else:
                                # 创建一个基本的JSON结构
                                data = {
                                    "video_id": video_id,
                                    "segment_id": segment_id,
                                    "text_content": vision_analysis[:500] + "..." if len(vision_analysis) > 500 else vision_analysis
                                }
                                detailed_logger.warning("无法从响应中提取JSON，创建基本结构")
                except json.JSONDecodeError as je:
                    detailed_logger.error(f"JSON解析失败: {str(je)}")
                    # 创建一个最小的JSON结构以便继续
                    data = {
                        "video_id": video_id,
                        "segment_id": segment_id,
                        "error": "JSON解析失败，无法提取分析结果",
                        "raw_content": vision_analysis[:500] + "..." if len(vision_analysis) > 500 else vision_analysis  # 限制长度
                    }
                    detailed_logger.warning("使用带有错误信息的占位符JSON")
                
                # 确保包含正确的segment_id信息
                if "segment_id" not in data or not data["segment_id"]:
                    data["segment_id"] = segment_id
                    detailed_logger.debug(f"添加缺失的segment_id: {segment_id}")
                
                # 添加到最终结果数组
                final_arr.append(data)
                detailed_logger.info(f"将分析结果添加到final_arr，当前长度: {len(final_arr)}")
                
                # 实时保存结果
                with open('actionSummary.json', 'w', encoding='utf-8') as f:
                    json.dump(final_arr, f, indent=4, ensure_ascii=False)
                detailed_logger.debug("已保存更新的actionSummary.json")
                
                # 计算和显示实时进度
                progress_percentage = (intervals_processed + 1) / num_intervals * 100
                progress_msg = f"\n=== 已完成: {progress_percentage:.1f}% [{intervals_processed + 1}/{num_intervals}] ==="
                print(progress_msg)
                detailed_logger.info(progress_msg)
                
            except Exception as e:
                error_msg = f"处理JSON结果失败: {str(e)}"
                print(error_msg)
                detailed_logger.error(error_msg)
                detailed_logger.error(traceback.format_exc())
                
                # 保存原始响应以便后续分析
                error_path = os.path.join(logs_dir, f'error_response_{video_id}_{segment_id}.txt')
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write(str(vision_analysis))
                
                # 记录错误详情
                detailed_logger.info(f"保存原始响应到 {error_path} 以便调查")
                
                # 添加到错误列表
                miss_arr.append(vision_analysis)
                detailed_logger.info("已将原始分析结果添加到miss_arr")
                
                # 创建一个错误信息对象添加到结果中，以便继续处理而不中断
                error_data = {
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "error": str(e),
                    "timestamp": time.time()
                }
                final_arr.append(error_data)
                detailed_logger.info("添加错误信息对象到final_arr以继续处理")

            spinner.stop()
            
        else:
            print(f"警告: 间隔 {segment_id} 没有捕获到帧，跳过")
        
        # 更新计数器和进度
        segment_counter += 1
        current_interval = interval_end
        intervals_processed += 1
        progress_bar.update(1)
    
    # 关闭资源
    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()
    
    print('视频提取、分析和转录完成！')
    
    # 生成以视频文件名命名的结果文件
    result_filename = f'actionSummary_{video_filename.split(".")[0]}.json'
    
    # 确保result文件夹存在
    if not os.path.exists('result'):
        os.makedirs('result')
    
    # 将完整结果保存到result文件夹中
    result_path = os.path.join('result', result_filename)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(final_arr, f, indent=4, ensure_ascii=False)
        
    # 保存最新的转录结果，但仅当有实际转录内容时才保存
    if global_transcript and global_transcript.strip():
        transcript_filename = f'transcript_{video_filename.split(".")[0]}.txt'
        transcript_path = os.path.join('result', transcript_filename)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(global_transcript)
        # 显示最终结果
        print(f"\n成功生成结果文件:")
        print(f"1. 分析结果: {result_path}")
        print(f"2. 转录文本: {transcript_path}")
    else:
        # 视频没有音频或转录内容为空
        print(f"\n成功生成结果文件:")
        print(f"1. 分析结果: {result_path}")
        print(f"2. 转录文本: 无 (视频没有音频)")
    
    return final_arr 

# 主程序入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='使用Gemini 1.5 Flash模型分析视频并生成动作摘要')
    parser.add_argument('--folder', type=str, default='DADA-2000-videos', help='视频文件夹路径')
    parser.add_argument('--interval', type=int, default=10, help='每个间隔的秒数')
    parser.add_argument('--frames', type=int, default=10, help='每个间隔的帧数')
    parser.add_argument('--single', type=str, default='', help='处理单个视频文件路径')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的视频数量')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已处理的视频')
    parser.add_argument('--start-at', type=int, default=0, help='从第几个视频开始处理(用于断点续传)')
    parser.add_argument('--retry-failed', action='store_true', help='尝试重新处理之前失败的视频')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help='日志记录级别')
    parser.add_argument('--save-interval', type=int, default=1, help='每处理多少个视频保存一次进度')
    parser.add_argument('--max-retries', type=int, default=3, help='API调用最大重试次数')
    parser.add_argument('--output-dir', type=str, default='result', help='输出目录')
    parser.add_argument('--processes', type=int, default=1, help='并行处理的进程数，默认为1 (注意：Gemini可能对并行请求有限制)')
    parser.add_argument('--speed-mode', action='store_true', help='启用速度优化模式，减少API调用和帧处理以提高速度')
    parser.add_argument('--video-list', type=str, default='', help='包含视频文件名列表的文件路径')
    parser.add_argument('--model', type=str, default='gemini-1.5-flash', help='使用的Gemini模型，默认为gemini-1.5-flash')
    
    args = parser.parse_args()
    
    # 设置Gemini模型
    if args.model:
        GEMINI_MODEL = args.model
        detailed_logger.info(f"使用命令行指定的Gemini模型: {GEMINI_MODEL}")
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level.upper())
    detailed_logger.setLevel(log_level)
    detailed_logger.info(f"设置日志级别为 {args.log_level}")
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        detailed_logger.info(f"创建输出目录: {args.output_dir}")
    
    # 处理单个视频
    if args.single:
        detailed_logger.info(f"处理单个视频: {args.single}")
        try:
            # 启动spinner显示处理进度
            spinner = Spinner(f"正在处理视频: {os.path.basename(args.single)}...", detailed_logger)
            spinner.start()
            
            # 处理视频
            AnalyzeVideo(args.single, args.interval, args.frames, args.speed_mode)
            
            # 停止spinner
            spinner.stop()
            print(f"成功处理视频: {os.path.basename(args.single)}")
            detailed_logger.info(f"成功处理视频: {args.single}")
        except Exception as e:
            # 确保spinner被停止
            try:
                spinner.stop()
            except:
                pass
                
            print(f"处理视频失败: {os.path.basename(args.single)}, 错误: {str(e)}")
            detailed_logger.error(f"处理视频失败: {args.single}, 错误: {str(e)}")
            detailed_logger.error(traceback.format_exc())
    else:
        # 从文件中读取视频列表
        if args.video_list and os.path.exists(args.video_list):
            video_folder = args.folder
            detailed_logger.info(f"从文件读取视频列表: {args.video_list}")
            video_files = []
            with open(args.video_list, 'r', encoding='utf-8') as f:
                for line in f:
                    video_name = line.strip()
                    if video_name:
                        video_path = os.path.join(video_folder, video_name)
                        video_files.append(video_path)
            
            print(f"从文件 {args.video_list} 中读取了 {len(video_files)} 个视频")
            detailed_logger.info(f"读取了 {len(video_files)} 个视频")
            
            # 处理每个视频
            successful = 0
            failed = 0
            skipped = 0
            for i, video_path in enumerate(video_files):
                video_name = os.path.basename(video_path)
                detailed_logger.info(f"处理视频 {i+1}/{len(video_files)}: {video_name}")
                
                # 检查视频是否存在
                if not os.path.exists(video_path):
                    print(f"视频文件不存在: {video_path}")
                    detailed_logger.error(f"视频文件不存在: {video_path}")
                    failed += 1
                    continue
                
                # 显示处理进度
                print(f"\n=== 处理视频 {i+1}/{len(video_files)}: {video_name} ===")
                
                # 检查是否已经处理过该视频
                result_filename = f'actionSummary_{video_name.split(".")[0]}.json'
                result_path = os.path.join(args.output_dir, result_filename)
                
                if os.path.exists(result_path) and not args.no_skip and not args.retry_failed:
                    print(f"视频 {video_name} 已经处理过，跳过")
                    detailed_logger.info(f"跳过已处理的视频: {video_name}")
                    skipped += 1
                    continue
                
                # 处理视频
                try:
                    AnalyzeVideo(video_path, args.interval, args.frames, args.speed_mode)
                    print(f"成功处理视频: {video_name}")
                    detailed_logger.info(f"成功处理视频: {video_name}")
                    successful += 1
                except Exception as e:
                    print(f"处理视频失败: {video_name}, 错误: {str(e)}")
                    detailed_logger.error(f"处理视频失败: {video_name}, 错误: {str(e)}")
                    detailed_logger.error(traceback.format_exc())
                    failed += 1
            
            print(f"\n处理完成: 成功 {successful} 个, 失败 {failed} 个, 跳过 {skipped} 个")
        else:
            print("你需要指定单个视频(--single)或视频列表文件(--video-list)，或者完整实现process_all_videos函数来批量处理所有视频")
            detailed_logger.warning("未指定单个视频或视频列表文件，也未完整实现批量处理函数") 