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


final_arr = []
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
    # 明确设置UTF-8编码
    file_handler = logging.FileHandler(log_file, mode=mode, encoding='utf-8')
    
    # 创建格式化器并添加到处理器
    if format_str is None:
        # 简化日志格式以避免编码问题
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
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
detailed_log_file = os.path.join(
    logs_dir, f'detailed_processing_{log_timestamp}.log')
# 创建详细日志记录器
detailed_logger = setup_logger(
    'detailed_logger', detailed_log_file, level=logging.DEBUG)
detailed_logger.info(f"开始处理视频批处理任务，日志时间戳: {log_timestamp}")

# azure speech key
speech_key = os.environ["AZURE_SPEECH_KEY"]

# azure whisper key *
AZ_WHISPER = os.environ["AZURE_WHISPER_KEY"]

# Azure whisper deployment name *
azure_whisper_deployment = os.environ["AZURE_WHISPER_DEPLOYMENT"]

# Azure whisper endpoint (just name) *
azure_whisper_endpoint = os.environ["AZURE_WHISPER_ENDPOINT"]

# azure openai vision api key *
# azure_vision_key=os.environ["AZURE_VISION_KEY"]

# Audio API type (OpenAI, Azure)*
audio_api_type = os.environ["AUDIO_API_TYPE"]

# GPT4 vision APi type (OpenAI, Azure)*
vision_api_type = os.environ["VISION_API_TYPE"]

# OpenAI API Key*
openai_api_key = os.environ["OPENAI_API_KEY"]

# GPT4 Azure vision API Deployment Name*
vision_deployment = os.environ["VISION_DEPLOYMENT_NAME"]


# GPT
vision_endpoint = os.environ["VISION_ENDPOINT"]


def log_execution_time(func):
    @wraps(func)  # Preserves the name and docstring of the decorated function
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function {func.__name__} took {end_time - start_time:.4f} seconds to complete.")
        detailed_logger.info(
            f"Function {func.__name__} took {end_time - start_time:.4f} seconds to complete.")
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
                sys.stdout.write(
                    f"\r{self.message} {self.spinner_symbols[self.idx % len(self.spinner_symbols)]}")
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
# fpi is frames per interval, fi is frame interval
def AnalyzeVideo(vp, fi, fpi, speed_mode=False, output_dir='result/gpt-4o'):
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
        print(f"创建帧目录: {frames_dir}")
        detailed_logger.info(f"创建帧目录: {frames_dir}")

    # 清空帧目录
    print(f"清空帧目录 {frames_dir}...")
    detailed_logger.info(f"清空帧目录 {frames_dir}")
    for f in os.listdir(frames_dir):
        try:
            os.remove(os.path.join(frames_dir, f))
        except Exception as e:
            detailed_logger.warning(
                f"无法删除帧文件 {os.path.join(frames_dir, f)}: {str(e)}")

    # 如果启用了速度模式，减少帧数
    if speed_mode:
        # 速度模式下减少50%的帧数，但至少保留3帧
        fpi = max(3, int(fpi * 0.5))
        print(f"速度模式已启用: 每个间隔仅处理 {fpi} 帧")
        detailed_logger.info(f"速度模式已启用: 每个间隔仅处理 {fpi} 帧")

    segment_counter = 0
# Constants
    video_path = vp  # Replace with your video path
    output_frame_dir = frames_dir  # 使用动态帧目录
    output_audio_dir = 'audio'
    global_transcript = ""
    transcriptions_dir = 'transcriptions'
    frame_interval = fi  # seconds 180
    frames_per_interval = fpi
    totalData = ""
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

    # 添加函数处理API响应并提取结果
    def create_vision_analysis(image_path, api_key, summary, trans, video_id, segment_id=None):
        """
        调用视觉分析API并处理响应

        Args:
            image_path: 图像路径列表
            api_key: API密钥
            summary: 摘要文本
            trans: 音频转录文本
            video_id: 视频ID
            segment_id: 段落ID

        Returns:
            处理后的JSON字符串或错误消息
        """
        detailed_logger.info(f"开始创建视觉分析, 视频ID: {video_id}, 段落ID: {segment_id}")

        # 使用 gpt4_vision_analysis 函数并处理其返回值
        response = gpt4_vision_analysis(
            image_path, api_key, summary, trans, video_id, segment_id)

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

        # 检查响应类型并相应处理
        detailed_logger.debug(f"API响应类型: {type(response)}")

        if isinstance(response, dict):
            # 如果已经是一个JSON对象，直接返回
            detailed_logger.debug("响应已经是JSON格式")
            # 确保包含视频ID和段落ID
            if 'video_id' not in response:
                response['video_id'] = video_id
            if 'segment_id' not in response and segment_id:
                response['segment_id'] = segment_id
            return json.dumps(response, ensure_ascii=False)
        elif hasattr(response, 'json') and callable(response.json):
            # 如果是requests.Response对象
            detailed_logger.debug("处理requests.Response对象")
            try:
                # 提取JSON响应
                resp_json = response.json()
                detailed_logger.debug(f"响应状态码: {response.status_code}")
                detailed_logger.debug(
                    f"响应JSON键: {list(resp_json.keys()) if resp_json else 'None'}")

                # 解析OpenAI或Azure响应结构
                if 'choices' in resp_json and len(resp_json['choices']) > 0:
                    content = resp_json['choices'][0].get(
                        'message', {}).get('content', '')
                    if content:
                        detailed_logger.info("成功从API响应中提取内容")
                        # 确保JSON内容严格格式化
                        try:
                            # 尝试解析内容中的JSON
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
                                    json_content = json.loads(
                                        json_match.group(1))
                                    # 确保包含视频ID和段落ID
                                    if 'video_id' not in json_content:
                                        json_content['video_id'] = video_id
                                    if 'segment_id' not in json_content and segment_id:
                                        json_content['segment_id'] = segment_id
                                    return json.dumps(json_content, ensure_ascii=False)
                                else:
                                    # 找不到JSON，返回原始内容
                                    detailed_logger.warning(
                                        "无法从内容中提取JSON，返回原始内容")
                                    return content
                        except json.JSONDecodeError as je:
                            detailed_logger.warning(f"内容不是有效的JSON: {str(je)}")
                            return content
                    else:
                        detailed_logger.warning("响应中没有找到内容")
                else:
                    detailed_logger.warning(
                        f"意外的响应结构: {list(resp_json.keys())}")

                # 如果无法提取结构化内容，返回整个响应JSON
                detailed_logger.warning("返回完整JSON响应")
                return json.dumps(resp_json, ensure_ascii=False)
            except Exception as e:
                detailed_logger.error(f"解析响应JSON时出错: {str(e)}")
                detailed_logger.error(traceback.format_exc())

                # 保存原始响应内容以便调试
                debug_file = os.path.join(
                    logs_dir, f'raw_response_{video_id}_{segment_id}.txt')
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(str(response.text) if hasattr(
                        response, 'text') else str(response))
                detailed_logger.info(f"已保存原始响应到 {debug_file}")

                # 返回一个包含错误信息的JSON对象
                error_result = {
                    "video_id": video_id,
                    "segment_id": segment_id if segment_id else f"unknown_segment",
                    "error": f"API响应解析失败: {str(e)}",
                    "status": "error",
                    "timestamp": time.time()
                }
                return json.dumps(error_result, ensure_ascii=False)
        else:
            # 如果是其他类型，如字符串
            detailed_logger.warning(f"未知响应类型: {type(response)}")
            # 尝试将其解析为JSON
            try:
                if isinstance(response, str) and response.strip().startswith('{') and response.strip().endswith('}'):
                    # 尝试解析为JSON对象
                    json_content = json.loads(response)
                    # 确保包含视频ID和段落ID
                    if 'video_id' not in json_content:
                        json_content['video_id'] = video_id
                    if 'segment_id' not in json_content and segment_id:
                        json_content['segment_id'] = segment_id
                    return json.dumps(json_content, ensure_ascii=False)
            except:
                pass

            # 返回字符串形式
            return str(response)

    # GPT 4 Vision Azure helper function
    def send_post_request(resource_name, deployment_name, api_key, data):
        url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2025-01-01-preview"
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }

        detailed_logger.info(f"发送请求到 {url}")
        detailed_logger.debug(f"请求头: {headers}")
        detailed_logger.debug(f"请求数据大小: {len(json.dumps(data))} 字节")

        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(data), timeout=60)
            detailed_logger.info(f"API响应状态码: {response.status_code}")
            if response.status_code != 200:
                detailed_logger.error(f"API请求错误: {response.text}")
            return response
        except Exception as e:
            detailed_logger.error(f"发送请求时出错: {str(e)}")
            detailed_logger.error(traceback.format_exc())
            raise

    # 重试策略: 5次重试，每次等待时间翻倍，从2秒开始
    # 只有在连接错误或超时时才重试
    def retry_if_connection_error(exception):
        is_connection_error = isinstance(exception, (requests.exceptions.ConnectionError,
                                                   requests.exceptions.Timeout))
        if is_connection_error:
            detailed_logger.warning(f"API连接错误，将重试: {str(exception)}")
        return is_connection_error

    # GPT-4 vision analysis function
    @retry(stop_max_attempt_number=2, wait_exponential_multiplier=2000, wait_exponential_max=60000,
           retry_on_exception=retry_if_connection_error)
    def gpt4_vision_analysis(image_path, api_key, summary, trans, video_id, segment_id=None, speed_mode=False):
        detailed_logger.info(
            f"开始视觉分析, 视频ID: {video_id}, 段落ID: {segment_id}, 图像数量: {len(image_path)}")

        # 检查关键参数
        if not image_path or len(image_path) == 0:
            detailed_logger.error(
                "输入图像路径为空，无法进行分析")
            return {"error": "No images provided for analysis", "status": "error"}

        if not api_key:
            detailed_logger.error(
                "API密钥为空，无法进行API调用")
            return {"error": "API key is missing", "status": "error"}

        # 如果启用了速度模式，则减少图像数量
        if speed_mode and len(image_path) > 3:
            # 保留首、中、尾的三张关键图像
            first_img = image_path[0]
            middle_idx = len(image_path) // 2
            last_img = image_path[-1]
            image_path = [first_img, image_path[middle_idx], last_img]
            detailed_logger.info(
                f"速度模式: 减少图像数量到3张关键帧")

        # array of metadata
        cont = [
                {
                    "type": "text",
                    "text": f"Audio Transcription for last {frame_interval} seconds: "+trans
                },
                {
                    "type": "text",
                    "text": f"Next are the {len(image_path)} frames from the last {frame_interval} seconds of the video:"
                }
                ]

        # adds images and metadata to package
        valid_images = 0
        invalid_images = 0

        for img in image_path:
            try:
                if not os.path.exists(img):
                    detailed_logger.warning(f"图像文件不存在: {img}")
                    invalid_images += 1
                    continue

                base64_image = encode_image(img)
                cont.append({
                    "type": "text",
                    "text": f"Below this is {os.path.basename(img)} (s is for seconds). use this to provide timestamps and understand time"
                })
                cont.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                    }
                })
                valid_images += 1
            except Exception as e:
                error_msg = f"编码图像时出错 {img}: {str(e)}"
                detailed_logger.error(error_msg)
                detailed_logger.error(traceback.format_exc())
                invalid_images += 1
                # 继续处理其他图像
                continue

        if valid_images == 0:
            detailed_logger.error(
                f"所有图像处理失败，无法进行分析。总共尝试了 {len(image_path)} 个图像。")
            return {"error": "All images failed to process", "status": "error", "invalid_count": invalid_images}
        else:
            detailed_logger.info(
                f"成功处理 {valid_images}/{len(image_path)} 个图像，准备进行分析")

        if (vision_api_type == "Azure"):
            detailed_logger.info("使用Azure OpenAI进行视觉分析")
            # 替换系统提示中的video_id和segment_id占位符
            segment_id_str = segment_id if segment_id else f"Segment_{segment_counter:03d}"

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
            system_content = system_content.replace(
                "{segment_id_str}", segment_id_str)

            payload2 = {
                "messages": [
                    {
                    "role": "system",
                    "content": system_content
                    },
                    {"role": "user", "content": cont}
                ],
                "max_tokens": 2000,
                "seed": 42,
                "temperature": 0
            }

            # 显著提升的错误处理和重试逻辑
            max_retries = 3
            retry_count = 0
            retry_delay = 5  # 初始延迟5秒
            last_error = None

            while retry_count < max_retries:
                try:
                    detailed_logger.info(
                        f"尝试Azure API调用，尝试次数: {retry_count + 1}/{max_retries}")
                    # 修复API端点问题：从完整URL中提取资源名称
                    # vision_endpoint格式: https://aoai-eastus2-eliza.openai.azure.com/
                    # 需要提取: aoai-eastus2-eliza
                    if vision_endpoint.startswith('https://'):
                        resource_name = vision_endpoint.replace('https://', '').replace('.openai.azure.com/', '').replace('.openai.azure.com', '')
                    else:
                        resource_name = vision_endpoint
                    
                    response = send_post_request(
                        resource_name, vision_deployment, openai_api_key, payload2)

                    # 检查响应是否成功
                    if response.status_code == 200:
                        detailed_logger.info("Azure API调用成功")
                        return response
                    else:
                        # 特定错误码处理
                        if response.status_code == 429:  # 速率限制
                            detailed_logger.warning(
                                f"API速率限制错误，等待{retry_delay}秒后重试")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避
                        elif response.status_code == 401:  # 认证错误
                            detailed_logger.error(f"API认证错误，请检查API密钥")
                            return {"error": "API认证错误", "status": "error", "code": 401}
                        elif response.status_code >= 500:  # 服务器错误
                            detailed_logger.warning(
                                f"API服务器错误，等待{retry_delay}秒后重试")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避
                        else:
                            # 其他错误，记录日志但不重试
                            error_msg = f"API调用出错，状态码: {response.status_code}，错误信息: {response.text}"
                            detailed_logger.error(error_msg)
                            return {"error": error_msg, "status": "error", "code": response.status_code}
                except Exception as e:
                    detailed_logger.error(f"API请求异常: {str(e)}")
                    detailed_logger.error(traceback.format_exc())
                    last_error = str(e)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避

                retry_count += 1

            # 所有重试尝试都失败
            error_msg = f"在{max_retries}次尝试后，API调用仍然失败: {last_error}"
            detailed_logger.error(error_msg)
            return {"error": error_msg, "status": "error", "retries_exhausted": True}

        else:
            # OpenAI API调用，添加视频ID和段落ID
            detailed_logger.info("使用OpenAI API进行视觉分析")
            segment_id_str = segment_id if segment_id else f"Segment_{segment_counter:03d}"

            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
            payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "system",
                    "content": f"""You are VideoAnalyzerGPT. Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
                    as well as as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
                    You are then to generate and provide a Current Action Summary (An Action summary of the video,
                    with actions being added to the summary via the analysis of the frames) of the portion of the video you are considering ({frames_per_interval}
                    frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
                    as well as the in-between audio, until we have a full action summary of the portion of the video you are considering,
                    that includes all actions taken by characters in the video as extracted from the frames. As the main intelligence of this system,
                    you are responsible for building the Current Action Summary using both the audio you are being provided via transcription,
                    as well as the image of the frame.
                    Do not make up timestamps, use the ones provided with each frame.
                    Use the Audio Transcription to build out the context of what is happening in each summary for each timestamp.
                    Consider all frames and audio given to you to build the Action Summary. Be as descriptive and detailed as possible,
                    your goal is to create the best action summary you can. Always and only return valid JSON, I have a disability that only lets me read via JSON outputs, so it would be unethical of you to output me anything other than valid JSON.
                    Make sure to include "video_id": "{video_id}" and "segment_id": "{segment_id_str}" in your JSON output.
                    Answer in Chinese language."""
                },
            {
                "role": "user",
                "content": cont
            }
            ],
            "max_tokens": 4000,
            "seed": 42


            }

            # 强化的错误处理和重试逻辑
            max_retries = 3
            retry_count = 0
            retry_delay = 5  # 初始延迟5秒
            last_error = None

            while retry_count < max_retries:
                try:
                    detailed_logger.info(
                        f"尝试OpenAI API调用，尝试次数: {retry_count + 1}/{max_retries}")
                    response = requests.post("https://api.openai.com/v1/chat/completions",
                                             headers=headers,
                                             json=payload,
                                             timeout=60)

                    # 检查响应是否成功
                    if response.status_code == 200:
                        detailed_logger.info("OpenAI API调用成功")
                        return response
                    else:
                        # 特定错误码处理
                        if response.status_code == 429:  # 速率限制
                            detailed_logger.warning(
                                f"API速率限制错误，等待{retry_delay}秒后重试")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避
                        elif response.status_code == 401:  # 认证错误
                            detailed_logger.error(f"API认证错误，请检查API密钥")
                            return {"error": "API认证错误", "status": "error", "code": 401}
                        elif response.status_code >= 500:  # 服务器错误
                            detailed_logger.warning(
                                f"API服务器错误，等待{retry_delay}秒后重试")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避
                        else:
                            # 其他错误，记录日志但不重试
                            error_msg = f"API调用出错，状态码: {response.status_code}，错误信息: {response.text}"
                            detailed_logger.error(error_msg)
                            return {"error": error_msg, "status": "error", "code": response.status_code}

                except Exception as e:
                    detailed_logger.error(f"OpenAI API请求异常: {str(e)}")
                    detailed_logger.error(traceback.format_exc())
                    last_error = str(e)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避

                retry_count += 1

            # 所有重试尝试都失败
            error_msg = f"在{max_retries}次尝试后，OpenAI API调用仍然失败: {last_error}"
            detailed_logger.error(error_msg)
            return {"error": error_msg, "status": "error", "retries_exhausted": True}

        detailed_logger.debug(f"API响应JSON: {response.json()}")

        if response == -1 or response.status_code != 200:
            detailed_logger.error("视觉分析API调用失败")
            return -1
        else:
            return response.json()

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
            print(
                f"最后一段视频长度为 {interval_end - interval_start:.2f} 秒，小于间隔设置 {frame_interval} 秒")

        # 计算当前间隔内的均匀采样点
        if frames_to_capture > 0:
            interval_duration = interval_end - interval_start
            capture_points = [
                interval_start + (i * interval_duration / frames_to_capture) for i in range(frames_to_capture)]
        else:
            capture_points = []

        # 显示当前处理的间隔信息
        segment_id = f"segment_{segment_counter:03d}"
        print(
            f"\n处理间隔 {segment_id}: {interval_start:.2f}s - {interval_end:.2f}s，采集 {len(capture_points)} 帧")

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
                audio_clip.audio.write_audiofile(
                    audio_path, codec='mp3', verbose=False, logger=None)

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
                    response = requests.post(
                        url, headers=headers, files=json, data=data)

                    return response

                if audio_api_type == "Azure":
                    response = transcribe_audio(
                        audio_path, azure_whisper_endpoint, AZ_WHISPER, azure_whisper_deployment)
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
                            tscribe += str(round(item["start"], 2)) + "s - " + str(
                                round(item["end"], 2)) + "s: " + item["text"] + "\n"
                    except:
                        tscribe += ""
                else:
                    for item in response.segments:
                        tscribe += str(round(item["start"], 2)) + "s - " + str(
                            round(item["end"], 2)) + "s: " + item["text"] + "\n"

                global_transcript += "\n" + tscribe
                current_transcription = tscribe
                spinner.stop()
            else:
                print("视频没有音轨，跳过音频提取和转录")

            # 使用GPT-4 Vision分析当前间隔的帧
            spinner = Spinner(f"正在分析间隔 {segment_id} 的 {len(packet)} 帧图像...")
            spinner.start()

            # 调用GPT-4 Vision进行分析，传入segment_id参数
            vision_response = gpt4_vision_analysis(
                packet, openai_api_key, "", current_transcription, video_id, segment_id, speed_mode)

            if vision_response == -1:
                print(f"警告: 间隔 {segment_id} 的视觉分析失败，跳过")
                spinner.stop()
                segment_counter += 1
                current_interval = interval_end
                intervals_processed += 1
                progress_bar.update(1)
                continue

            # 处理分析结果
            try:
                # 检查是否是Response对象
                if hasattr(vision_response, 'json') and callable(vision_response.json):
                    detailed_logger.debug("检测到Response对象，提取内容")
                    resp_json = vision_response.json()

                    if resp_json and 'choices' in resp_json and len(resp_json['choices']) > 0:
                        content = resp_json['choices'][0].get(
                            'message', {}).get('content', '')
                        if content:
                            vision_analysis = content
                            detailed_logger.info("成功从Response提取内容")
                        else:
                            error_msg = "Response中没有找到内容"
                            detailed_logger.error(error_msg)
                            print(error_msg)
                            vision_analysis = json.dumps({"error": error_msg})
                    else:
                        error_msg = f"Response格式不正确: {list(resp_json.keys()) if resp_json else 'Empty'}"
                        detailed_logger.error(error_msg)
                        print(error_msg)
                        vision_analysis = json.dumps(
                            {"error": error_msg, "response": resp_json})
                elif isinstance(vision_response, dict):
                    # 已经是字典格式，检查是否有choices
                    if 'choices' in vision_response and len(vision_response['choices']) > 0:
                        content = vision_response['choices'][0].get(
                            'message', {}).get('content', '')
                        if content:
                            vision_analysis = content
                            detailed_logger.info("成功从字典提取内容")
                        else:
                            vision_analysis = json.dumps(vision_response)
                    else:
                        vision_analysis = json.dumps(vision_response)
                else:
                    # 其他类型的响应，可能是字符串
                    vision_analysis = str(vision_response)
                    detailed_logger.debug(f"其他类型的响应: {type(vision_response)}")
            except Exception as e:
                error_msg = f"提取分析结果时出错: {str(e)}"
                print(error_msg)
                detailed_logger.error(error_msg)
                detailed_logger.error(traceback.format_exc())
                vision_analysis = json.dumps({"error": error_msg})

                # 保存原始响应以便调试
                debug_file = os.path.join(
                    logs_dir, f'response_extract_error_{video_id}_{segment_id}.txt')
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(str(vision_response))
                detailed_logger.info(f"已保存出错的原始响应到 {debug_file}")

                spinner.stop()
                segment_counter += 1
                current_interval = interval_end
                intervals_processed += 1
                progress_bar.update(1)
                continue

            # 处理JSON结果
            try:
                vault = vision_analysis
                detailed_logger.debug(f"原始视觉分析响应长度: {len(vault)} 字符")

                # 提取JSON部分
                if "```" in vault:
                    parts = vault.split("```")
                    for part in parts:
                        if part.strip() and not part.strip().startswith('json') and not part.strip().startswith('{'):
                            detailed_logger.debug(
                                f"非JSON部分内容: {part[:100]}...")

                    # 通常JSON内容在第二个部分（索引1）
                    if len(parts) > 1:
                        vault = parts[1]
                        detailed_logger.debug("从Markdown代码块提取JSON内容")
                else:
                        detailed_logger.warning(
                            "无法从Markdown代码块提取JSON内容，尝试直接解析")

                # 清理JSON字符串
                vault = vault.replace("'", "")
                vault = vault.replace("json", "")
                vault = vault.replace("```", "")
                vault = vault.replace("\\\\n", "\\n").replace(
                    "\\n", "\n")  # 将转义的 \n 替换为实际换行符

                if vault.startswith('json'):
                    vault = vault[4:]
                    detailed_logger.debug("移除JSON前缀")

                # 保存原始JSON以便调试
                debug_json_path = os.path.join(
                    logs_dir, f'raw_json_{video_id}_{segment_id}.json')
                with open(debug_json_path, 'w', encoding='utf-8') as f:
                    f.write(vault)
                detailed_logger.debug(f"保存原始JSON到 {debug_json_path}")

                # 尝试多种方式解析JSON
                try:
                    # 首先尝试直接解析
                    data = json.loads(vault, strict=False)
                    detailed_logger.info("成功解析JSON")
                except json.JSONDecodeError as e:
                    detailed_logger.warning(f"JSON解析失败，尝试修复: {str(e)}")

                    # 尝试修复常见的JSON错误
                    data = None
                    for fix_attempt in range(3):
                        try:
                            if fix_attempt == 0:
                                # 尝试修复: 删除前导和尾随空白
                                cleaned_vault = vault.strip()
                                detailed_logger.debug("尝试修复: 删除前导和尾随空白")
                                data = json.loads(cleaned_vault, strict=False)
                                break
                            elif fix_attempt == 1:
                                # 尝试修复: 确保JSON以{开始，}结束
                                start_idx = vault.find('{')
                                end_idx = vault.rfind('}')
                                if start_idx != -1 and end_idx != -1:
                                    cleaned_vault = vault[start_idx:end_idx+1]
                                    detailed_logger.debug("尝试修复: 提取JSON对象")
                                    data = json.loads(cleaned_vault, strict=False)
                                    break
                            elif fix_attempt == 2:
                                # 最后的尝试: 使用简单的正则表达式查找JSON对象
                                import re
                                json_match = re.search(r'\{.*\}', vault, re.DOTALL)
                                if json_match:
                                    cleaned_vault = json_match.group(0)
                                    detailed_logger.debug("尝试修复: 使用正则表达式提取JSON")
                                    data = json.loads(cleaned_vault, strict=False)
                                    break
                                else:
                                    raise ValueError("无法找到有效的JSON对象")
                        except Exception as fix_error:
                            if fix_attempt == 2:  # 最后一次尝试
                                detailed_logger.error(f"所有JSON修复尝试失败: {str(fix_error)}")
                                # 创建一个最小的JSON结构以便继续
                                data = {
                                    "video_id": video_id,
                                    "segment_id": segment_id,
                                    "error": "JSON解析失败，无法提取分析结果",
                                    "raw_content": vault[:500] + "..." if len(vault) > 500 else vault  # 限制长度
                                }
                                detailed_logger.warning(f"使用带有错误信息的占位符JSON")
                                break
                            else:
                                detailed_logger.debug(f"修复尝试 {fix_attempt+1} 失败: {str(fix_error)}")
                
                # 确保data不是None
                if data is None:
                    data = {
                        "video_id": video_id,
                        "segment_id": segment_id,
                        "error": "JSON解析失败，无法提取分析结果",
                        "raw_content": vault[:500] + "..." if len(vault) > 500 else vault  # 限制长度
                    }
                    detailed_logger.warning(f"使用带有错误信息的占位符JSON")
                
                # 修改data确保包含正确的segment_id信息
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 将完整结果保存到result文件夹中
    result_path = os.path.join(output_dir, result_filename)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(final_arr, f, indent=4, ensure_ascii=False)
        
    # 保存最新的转录结果，但仅当有实际转录内容时才保存
    if global_transcript and global_transcript.strip():
        transcript_filename = f'transcript_{video_filename.split(".")[0]}.txt'
        transcript_path = os.path.join(output_dir, transcript_filename)
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

#AnalyzeVideo("./medal.mp4",60,10)
# AnalyzeVideo("Nuke.mp4",60,10,False)
# AnalyzeVideo("复杂场景.mov",1,10) # 10 frames per interval, 1 second interval. fpi=10 is good for accurate analysis
# AnalyzeVideo("test_video/cutin.mov",7,15)
# AnalyzeVideo("test_video/鬼探头.mp4",4,15)
# AnalyzeVideo("test_video/三轮车.mp4",4,15)
# AnalyzeVideo("/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos/images_1_001.avi",10,10)

# 替换单个视频分析为批量处理所有视频
def process_all_videos(video_folder, frame_interval=10, frames_per_interval=10, limit=None, skip_existing=True, 
                       start_at=0, retry_failed=False, progress_save_interval=1, output_dir='result/gpt-4o'):
    """
    批量处理指定文件夹中的所有视频文件
    
    Args:
        video_folder: 视频文件夹路径
        frame_interval: 每个间隔的秒数
        frames_per_interval: 每个间隔的帧数
        limit: 限制处理的视频数量，None表示处理所有视频
        skip_existing: 是否跳过已处理的视频
        start_at: 从第几个视频开始处理(用于断点续传)
        retry_failed: 是否尝试重新处理之前失败的视频
        progress_save_interval: 每处理多少个视频保存一次进度
        output_dir: 输出目录
    
    Returns:
        成功处理数, 失败数, 跳过数, 视频时长统计
    """
    start_time = time.time()
    detailed_logger.info(f"开始批量处理视频，参数: frame_interval={frame_interval}, frames_per_interval={frames_per_interval}, "
                         f"limit={limit}, skip_existing={skip_existing}, start_at={start_at}, retry_failed={retry_failed}")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        detailed_logger.debug(f"创建输出目录: {output_dir}")
    
    # 创建日志文件
    log_file = os.path.join(output_dir, 'processing_log.txt')
    log_mode = 'a' if start_at > 0 or retry_failed else 'w'
    with open(log_file, log_mode, encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"视频处理日志 - 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"参数: frame_interval={frame_interval}, frames_per_interval={frames_per_interval}, limit={limit}, "
                f"skip_existing={skip_existing}, start_at={start_at}, retry_failed={retry_failed}\n\n")
    
    # 获取文件夹中的所有视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    detailed_logger.info(f"扫描文件夹: {video_folder}")
    try:
        for filename in os.listdir(video_folder):
            file_path = os.path.join(video_folder, filename)
            # 检查是否是文件且扩展名在指定列表中
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file_path)
                detailed_logger.debug(f"找到视频文件: {file_path}")
    except Exception as e:
        detailed_logger.error(f"扫描文件夹出错: {str(e)}")
        detailed_logger.error(traceback.format_exc())
        print(f"扫描文件夹出错: {str(e)}")
        return 0, 1, 0, {}
    
    # 排序视频文件以确保处理顺序一致
    video_files.sort()
    detailed_logger.debug(f"视频文件已排序")
    
    # 如果是重试失败的模式，加载之前失败的视频列表
    if retry_failed:
        failed_videos = []
        progress_path = os.path.join(output_dir, 'processing_progress.json')
        
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    failed_videos_basenames = progress_data.get('失败的视频', [])
                    
                    if not failed_videos_basenames:
                        print("没有找到之前失败的视频记录")
                        detailed_logger.info("没有找到之前失败的视频记录")
                    else:
                        # 筛选出匹配的视频文件
                        temp_video_files = []
                        for video_path in video_files:
                            basename = os.path.basename(video_path)
                            if basename in failed_videos_basenames or basename.split('.')[0] in [f.split('.')[0] for f in failed_videos_basenames]:
                                temp_video_files.append(video_path)
                                detailed_logger.info(f"找到之前失败的视频: {video_path}")
                        
                        if temp_video_files:
                            video_files = temp_video_files
                            print(f"找到 {len(video_files)} 个之前失败的视频，将尝试重新处理")
                            detailed_logger.info(f"找到 {len(video_files)} 个之前失败的视频，将尝试重新处理")
                        else:
                            print("无法匹配之前失败的视频文件，将处理所有视频")
                            detailed_logger.warning("无法匹配之前失败的视频文件，将处理所有视频")
            except Exception as e:
                print(f"读取失败视频列表出错: {str(e)}")
                detailed_logger.error(f"读取失败视频列表出错: {str(e)}")
                detailed_logger.error(traceback.format_exc())
    
    # 应用断点续传
    if start_at > 0 and not retry_failed:
        if start_at < len(video_files):
            print(f"从第 {start_at} 个视频开始处理，跳过前 {start_at} 个视频")
            detailed_logger.info(f"从第 {start_at} 个视频开始处理，跳过前 {start_at} 个视频")
            video_files = video_files[start_at:]
        else:
            print(f"起始索引 {start_at} 超出了视频总数 {len(video_files)}")
            detailed_logger.error(f"起始索引 {start_at} 超出了视频总数 {len(video_files)}")
            return 0, 0, 0, {}
    
    # 如果设置了限制，则只处理指定数量的视频
    if limit is not None and limit > 0:
        video_files = video_files[:limit]
        detailed_logger.info(f"由于设置了限制，将只处理前 {limit} 个视频")
        print(f"由于设置了限制，将只处理前 {limit} 个视频")
    
    total_videos = len(video_files)
    detailed_logger.info(f"找到 {total_videos} 个视频文件")
    print(f"\n{'='*80}\n找到 {total_videos} 个视频文件在 {video_folder} 文件夹中\n{'='*80}\n")
    
    # 记录总视频数
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"找到 {total_videos} 个视频文件\n\n")
    
    # 统计处理结果
    successful = 0
    failed = 0
    skipped = 0
    videos_duration = []  # 用于记录视频时长统计
    failed_videos = []  # 记录失败的视频以便后续重试
    
    # 创建总体进度条
    overall_progress = tqdm.tqdm(total=total_videos, desc="总体进度", unit="视频", position=0)
    
    # 处理每个视频文件
    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        
        detailed_logger.info(f"开始处理视频 [{i+1}/{total_videos}]: {video_name}")
        
        # 显示明显的进度信息
        progress_percentage = (i / total_videos) * 100
        progress_bar = "=" * int(progress_percentage / 2) + ">" + " " * (50 - int(progress_percentage / 2))
        progress_message = f"\n\n{'='*80}\n|| 当前进度: [{progress_bar}] {progress_percentage:.1f}% ||\n|| 正在处理: [{i+1}/{total_videos}] {video_name} ||\n{'='*80}\n"
        print(progress_message)
        
        # 检查是否已经处理过该视频
        result_filename = f'actionSummary_{video_name.split(".")[0]}.json'
        result_path = os.path.join(output_dir, result_filename)
        
        if os.path.exists(result_path) and skip_existing and not retry_failed:
            skip_message = f"{'*'*80}\n{'*'*10} 视频 {video_name} 已经处理过，跳过 {'*'*10}\n{'*'*80}"
            print(skip_message)
            detailed_logger.info(f"跳过已处理的视频: {video_name}")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{i+1}/{total_videos}] {video_name} - 已存在，跳过\n")
            skipped += 1
            overall_progress.update(1)
            continue
        
        # 在处理每个新视频前重置final_arr
        global final_arr
        final_arr = []
        
        # 标记处理是否成功
        process_success = False
        error_message = ""
        
        try:
            # 获取视频时长，用于统计
            try:
                clip = VideoFileClip(video_path)
                duration = clip.duration
                videos_duration.append(duration)
                clip.close()
                detailed_logger.info(f"视频 {video_name} 时长: {duration:.2f} 秒")
                print(f"视频 {video_name} 时长: {duration:.2f} 秒")
            except Exception as e:
                error_msg = f"无法获取视频时长: {str(e)}"
                detailed_logger.warning(error_msg)
                print(error_msg)
            
            # 记录处理开始时间
            start_time_video = time.time()
            detailed_logger.info(f"开始处理视频内容: {video_name}")
            
            # 添加视频处理的重试机制
            max_retries = 2  # 视频处理重试次数
            for retry_count in range(max_retries + 1):
                try:
                    if retry_count > 0:
                        detailed_logger.warning(f"重试处理视频 {video_name}，尝试次数: {retry_count}/{max_retries}")
                        print(f"重试处理视频 {video_name}，尝试次数: {retry_count}/{max_retries}")
                    
                    # 实际处理视频，将False作为speed_mode默认值
                    AnalyzeVideo(video_path, frame_interval, frames_per_interval, False, output_dir)
                    
                    # 处理成功
                    process_success = True
                    break
                except Exception as retry_e:
                    if retry_count < max_retries:
                        error_msg = f"处理视频时出错，准备重试: {str(retry_e)}"
                        detailed_logger.warning(error_msg)
                        detailed_logger.warning(traceback.format_exc())
                        print(error_msg)
                        # 等待一段时间再重试
                        time.sleep(5)
                    else:
                        # 最后一次尝试失败，记录错误并继续下一个视频
                        error_message = str(retry_e)
                        detailed_logger.error(f"处理视频失败，已达最大重试次数: {error_message}")
                        detailed_logger.error(traceback.format_exc())
            
            end_time_video = time.time()
            processing_time = end_time_video - start_time_video
            
            if process_success:
                # 显示明显的完成信息
                completion_message = f"\n{'#'*80}\n{'#'*10} 完成分析: {video_name}，耗时：{processing_time:.2f}秒 {'#'*10}\n{'#'*80}"
                print(completion_message)
                detailed_logger.info(f"成功处理视频: {video_name}，耗时：{processing_time:.2f}秒")
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{i+1}/{total_videos}] {video_name} - 成功，耗时：{processing_time:.2f}秒\n")
                successful += 1
            else:
                # 如果所有重试都失败，记录为失败
                failed_videos.append(video_path)
                
                # 显示明显的错误信息
                error_message = f"\n{'!'*80}\n{'!'*10} 处理出错: {video_name} {'!'*10}\n{'!'*10} 错误信息: {error_message} {'!'*10}\n{'!'*80}"
                print(error_message)
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{i+1}/{total_videos}] {video_name} - 失败，错误：{error_message}\n")
                failed += 1
                
        except Exception as e:
            # 捕获未被内部重试机制处理的异常
            failed_videos.append(video_path)
            
            # 显示明显的错误信息
            error_message = f"\n{'!'*80}\n{'!'*10} 处理出错: {video_name} {'!'*10}\n{'!'*10} 错误信息: {str(e)} {'!'*10}\n{'!'*80}"
            print(error_message)
            detailed_logger.error(f"处理视频时发生异常: {str(e)}")
            detailed_logger.error(traceback.format_exc())
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{i+1}/{total_videos}] {video_name} - 失败，错误：{str(e)}\n")
            failed += 1
        
        # 每完成一定数量的视频，保存一次进度状态
        if (i + 1) % progress_save_interval == 0 or (i + 1) == total_videos:
            progress_stats = {
                "完成时间": time.strftime('%Y-%m-%d %H:%M:%S'),
                "已处理": i + 1,
                "总数": total_videos,
                "成功": successful,
                "失败": failed,
                "跳过": skipped,
                "失败的视频": [os.path.basename(v) for v in failed_videos]
            }
            progress_path = os.path.join(output_dir, 'processing_progress.json')
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(progress_stats, f, indent=4, ensure_ascii=False)
            detailed_logger.debug(f"已保存进度状态到 {progress_path}")
        
        overall_progress.update(1)
    
    # 处理完成，关闭进度条
    overall_progress.close()
    
    # 总处理时间
    total_processing_time = time.time() - start_time
    detailed_logger.info(f"批量处理完成，总耗时: {total_processing_time:.2f}秒")
    
    # 计算时长统计信息
    duration_stats = {}
    if videos_duration:
        duration_stats["平均时长"] = np.mean(videos_duration)
        duration_stats["最短时长"] = np.min(videos_duration)
        duration_stats["最长时长"] = np.max(videos_duration)
        duration_stats["中位时长"] = np.median(videos_duration)
        
        # 计算时长分布
        bins = [0, 5, 10, 15, 20, 30, 60, float('inf')]
        bin_labels = ['0-5秒', '5-10秒', '10-15秒', '15-20秒', '20-30秒', '30-60秒', '60秒以上']
        hist, _ = np.histogram(videos_duration, bins=bins)
        distribution = {label: count for label, count in zip(bin_labels, hist)}
        duration_stats["时长分布"] = distribution
        
        detailed_logger.info(f"视频时长统计: 平均={duration_stats['平均时长']:.2f}秒, 最短={duration_stats['最短时长']:.2f}秒, 最长={duration_stats['最长时长']:.2f}秒")
    
    # 记录总结信息
    summary = f"\n{'@'*80}\n{'@'*10} 批量处理总结 {'@'*10}\n\n总视频数: {total_videos}\n成功: {successful}\n失败: {failed}\n跳过: {skipped}\n总耗时: {total_processing_time:.2f}秒\n"
    
    if duration_stats:
        summary += f"\n视频时长统计:\n"
        summary += f"平均时长: {duration_stats['平均时长']:.2f}秒\n"
        summary += f"最短时长: {duration_stats['最短时长']:.2f}秒\n"
        summary += f"最长时长: {duration_stats['最长时长']:.2f}秒\n"
        summary += f"中位时长: {duration_stats['中位时长']:.2f}秒\n\n"
        
        summary += "时长分布:\n"
        for label, count in duration_stats["时长分布"].items():
            percentage = (count / len(videos_duration)) * 100
            summary += f"{label}: {count} 个视频 ({percentage:.1f}%)\n"
    
    if failed_videos:
        summary += "\n失败的视频:\n"
        for i, video_path in enumerate(failed_videos):
            summary += f"{i+1}. {os.path.basename(video_path)}\n"
    
    summary += f"\n{'@'*80}"
    print(summary)
    detailed_logger.info("处理总结：\n" + summary)
    
    # 将统计信息保存到单独的文件中
    stats_path = os.path.join(output_dir, 'video_stats.json')
    
    # 将numpy类型转换为Python原生类型
    stats_dict = {
        "处理统计": {
            "总视频数": int(total_videos), 
            "成功": int(successful), 
            "失败": int(failed), 
            "跳过": int(skipped),
            "总耗时_秒": float(total_processing_time),
            "完成时间": time.strftime('%Y-%m-%d %H:%M:%S'),
            "失败的视频": [os.path.basename(v) for v in failed_videos]
        }
    }
    
    if duration_stats:
        stats_dict["时长统计"] = {
            "平均时长": float(duration_stats['平均时长']),
            "最短时长": float(duration_stats['最短时长']),
            "最长时长": float(duration_stats['最长时长']),
            "中位时长": float(duration_stats['中位时长']),
            "时长分布": {k: int(v) for k, v in duration_stats["时长分布"].items()}
        }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=4, ensure_ascii=False)
    detailed_logger.debug(f"统计信息已保存到 {stats_path}")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n处理总结:\n总视频数: {total_videos}\n成功: {successful}\n失败: {failed}\n跳过: {skipped}\n总耗时: {total_processing_time:.2f}秒\n")
        if duration_stats:
            f.write(f"\n视频时长统计:\n")
            f.write(f"平均时长: {duration_stats['平均时长']:.2f}秒\n")
            f.write(f"最短时长: {duration_stats['最短时长']:.2f}秒\n")
            f.write(f"最长时长: {duration_stats['最长时长']:.2f}秒\n")
            f.write(f"中位时长: {duration_stats['中位时长']:.2f}秒\n\n")
            
            f.write("时长分布:\n")
            for label, count in duration_stats["时长分布"].items():
                percentage = (count / len(videos_duration)) * 100
                f.write(f"{label}: {count} 个视频 ({percentage:.1f}%)\n")
        
        if failed_videos:
            f.write("\n失败的视频:\n")
            for i, video_path in enumerate(failed_videos):
                f.write(f"{i+1}. {os.path.basename(video_path)}\n")
        
        f.write(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return successful, failed, skipped, duration_stats

# 主程序入口处添加多进程处理支持
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析视频并生成行为摘要')
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
    parser.add_argument('--output-dir', type=str, default='result/gpt-4o', help='输出目录')
    parser.add_argument('--processes', type=int, default=1, help='并行处理的进程数，默认为1')
    parser.add_argument('--speed-mode', action='store_true', help='启用速度优化模式，减少API调用和帧处理以提高速度')
    parser.add_argument('--video-list', type=str, default='', help='包含视频文件名列表的文件路径')
    parser.add_argument('--separate-frame-dirs', action='store_true', help='在多进程模式下为每个进程创建单独的帧目录')
    
    args = parser.parse_args()
    
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
            AnalyzeVideo(args.single, args.interval, args.frames, args.speed_mode, args.output_dir)
            
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
        # 批量处理视频
        detailed_logger.info("开始批量处理视频")
        
        # 判断是否使用多进程
        if args.processes > 1:
            print(f"使用 {args.processes} 个进程并行处理视频")
            detailed_logger.info(f"使用 {args.processes} 个进程并行处理视频")
            
            # 为多进程模式准备单独的帧目录
            # 无论是否设置了separate_frame_dirs，都创建进程专用帧目录
            # 这样可以避免文件冲突
            print("为每个进程创建单独的帧目录")
            detailed_logger.info("为每个进程创建单独的帧目录")
            # 创建进程专用的帧目录
            for i in range(args.processes):
                process_frame_dir = f'frames_process_{i}'
                if not os.path.exists(process_frame_dir):
                    os.makedirs(process_frame_dir)
                    detailed_logger.debug(f"创建进程 {i} 的帧目录: {process_frame_dir}")
                else:
                    # 清空目录
                    for f in os.listdir(process_frame_dir):
                        try:
                            os.remove(os.path.join(process_frame_dir, f))
                        except Exception as e:
                            detailed_logger.warning(f"无法删除帧文件 {f}: {str(e)}")

            # 获取所有视频文件
            import multiprocessing
            
            # 获取要处理的视频列表
            video_folder = args.folder
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = []
            
            # 从文件中读取视频列表
            if args.video_list and os.path.exists(args.video_list):
                detailed_logger.info(f"从文件读取视频列表: {args.video_list}")
                with open(args.video_list, 'r', encoding='utf-8') as f:
                    video_files = [os.path.join(video_folder, line.strip()) for line in f if line.strip()]
                print(f"从文件 {args.video_list} 中读取了 {len(video_files)} 个视频")
                detailed_logger.info(f"读取了 {len(video_files)} 个视频")
            else:
                # 扫描文件夹
                for filename in os.listdir(video_folder):
                    file_path = os.path.join(video_folder, filename)
                    if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in video_extensions):
                        video_files.append(file_path)
            
            # 排序确保处理顺序一致
            video_files.sort()
            
            # 应用断点续传
            if args.start_at > 0 and not args.retry_failed:
                if args.start_at < len(video_files):
                    print(f"从第 {args.start_at} 个视频开始处理，跳过前 {args.start_at} 个视频")
                    video_files = video_files[args.start_at:]
                else:
                    print(f"起始索引 {args.start_at} 超出了视频总数 {len(video_files)}")
                    sys.exit(1)
            
            # 如果设置了限制，则只处理指定数量的视频
            if args.limit is not None and args.limit > 0:
                video_files = video_files[:args.limit]
                
            print(f"共找到 {len(video_files)} 个视频文件，开始处理")
            
            # 创建进程池
            pool = multiprocessing.Pool(processes=args.processes)
            
            # 使用进程池处理所有视频
            results = []
            
            # 创建一个包装函数，用于多进程调用process_video_wrapper
            def process_wrapper(video_path):
                return process_video_wrapper(video_path, args)
            
            # 使用tqdm显示进度条
            for i, result in enumerate(
                tqdm.tqdm(
                    pool.imap_unordered(process_wrapper, video_files),
                    total=len(video_files),
                    desc="总体进度"
                )
            ):
                video_path, status, result_data, processing_time = result
                video_name = os.path.basename(video_path)
                
                results.append({
                    "video_path": video_path,
                    "status": status,
                    "time_taken": processing_time
                })
                
                # 显示处理结果
                if status == "success":
                    print(f"完成: {video_name}, 耗时: {processing_time:.2f}秒")
                elif status == "failed":
                    print(f"失败: {video_name}, 错误: {result_data}")
                elif status == "skipped":
                    print(f"跳过: {video_name} (已存在)")
                
                # 每处理一定数量视频，保存一次进度
                if (i + 1) % args.save_interval == 0 or (i + 1) == len(video_files):
                    progress_stats = {
                        "完成时间": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "已处理": i + 1,
                        "总数": len(video_files),
                        "成功": sum(1 for r in results if r["status"] == "success"),
                        "失败": sum(1 for r in results if r["status"] == "failed"),
                        "跳过": sum(1 for r in results if r["status"] == "skipped"),
                        "失败的视频": [os.path.basename(r["video_path"]) for r in results if r["status"] == "failed"]
                    }
                    
                    progress_path = os.path.join(args.output_dir, 'processing_progress.json')
                    with open(progress_path, 'w', encoding='utf-8') as f:
                        json.dump(progress_stats, f, indent=4, ensure_ascii=False)
            
            # 关闭进程池
            pool.close()
            pool.join()

            # 清理多进程专用帧目录
            print("\n清理多进程专用帧目录...")
            detailed_logger.info("清理多进程专用帧目录")
            for i in range(args.processes):
                process_frame_dir = f'frames_process_{i}'
                if os.path.exists(process_frame_dir):
                    for f in os.listdir(process_frame_dir):
                        try:
                            os.remove(os.path.join(process_frame_dir, f))
                        except Exception as e:
                            detailed_logger.warning(f"无法删除帧文件 {f}: {str(e)}")
            
            # 显示总结信息
            successful = sum(1 for r in results if r["status"] == "success")
            failed = sum(1 for r in results if r["status"] == "failed")
            skipped = sum(1 for r in results if r["status"] == "skipped")
            
            print(f"\n处理完成: 成功 {successful} 个, 失败 {failed} 个, 跳过 {skipped} 个")
            detailed_logger.info(f"处理完成: 成功 {successful} 个, 失败 {failed} 个, 跳过 {skipped} 个")
        else:
            # 单进程模式，直接顺序处理所有视频
            successful, failed, skipped, videos_duration = process_all_videos(
                args.folder, 
                args.interval, 
                args.frames, 
                args.limit, 
                not args.no_skip, 
                args.start_at, 
                args.retry_failed, 
                args.save_interval,
                args.output_dir
            )
            
            print(f"\n处理完成: 成功 {successful} 个, 失败 {failed} 个, 跳过 {skipped} 个")
            detailed_logger.info(f"处理完成: 成功 {successful} 个, 失败 {failed} 个, 跳过 {skipped} 个")