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
# 修复导入方式
try:
    # 旧版本的导入方式
    from google.genai import types
except ImportError:
    # 新版本的导入方式
    from google.generativeai import types

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
detailed_log_file = os.path.join(logs_dir, f'gemini_processing_{log_timestamp}.log')
# 创建详细日志记录器
detailed_logger = setup_logger('detailed_logger', detailed_log_file, level=logging.DEBUG)
detailed_logger.info(f"开始处理视频批处理任务，时间戳: {log_timestamp}")

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
    # 使用新版API初始化
    genai.configure(api_key=GEMINI_API_KEY)
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
def AnalyzeVideo(vp, fi, fpi, speed_mode=False, output_dir='result/gemini-testinterval'):  #fpi is frames per interval, fi is frame interval
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
    for directory in [output_frame_dir, output_audio_dir, transcriptions_dir, output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Encode image to base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

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
                image_obj = {"mime_type": "image/jpeg", "data": image_bytes}
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

**IMPORTANT: Accuracy Requirements**
1. Weather Conditions: ONLY describe weather conditions (rain, snow, fog, etc.) if they are clearly visible in the frames. Never assume or guess weather conditions. If you cannot clearly determine the weather, simply describe the scene as "daytime" or "nighttime" without making weather assumptions.
2. Pedestrian Descriptions: Be precise about the number and characteristics of pedestrians. Avoid vague terms like "a group" unless necessary. Count and describe individual pedestrians when possible.
3. Risk Assessment: Only label actions as "ghost probing", "cut-in", or other risk categories when the behavior clearly and unambiguously matches the definition. When in doubt, label as "none" rather than over-reporting risks.
4. Evidence-Based Descriptions: Every statement in your analysis must be directly supported by visual evidence in the frames. Avoid conjecture or assumptions about things not directly visible.

**Task 1: Identify and Predict potential "Ghost Probing(专业术语：鬼探头)",Cut-in(加塞),left/right-side overtaken(左侧/右侧他车对自车超车) etc Behavior**
1)"Ghost Probing" behavior refers to a person suddenly darting out from either left or right side of the car itself and also from BEHIND an object that blocks the driver's view, such as a parked car, a tree, or a board, directly dash into the driver's path. 
Note that: only those coming from visual blind spot of the car itself can be considered as 鬼探头. b) cut-in加塞 is different from 鬼探头       
Ghost Probing:
    Core Definition: When a pedestrian, cyclist, or another vehicle suddenly emerges from a blind spot e.g., behind an obstacle in front of the driver's path, leaving the driver no time to react.
    This behavior usually occurs when individuals, either not paying attention to the traffic conditions or in a hurry, suddenly obstruct the driver's view, creating an emergency situation with very little reaction time. This can easily lead to traffic accidents.
    通常指行人或车辆从视觉盲区（如车辆前方的遮挡物后面）突然冒出，导致驾驶员来不及反应. 
    Characteristics:
        Sudden emergence of from a visual blind spot.
        or, even without a blind spot, it involves an unexpected event, leaving no reaction time for the driver.

    Extended Cases for Ghost Probing:
    1.Sudden Close Proximity of a Moving Object:
    Description: A moving object rapidly approaches the self-driving vehicle, even though it does not emerge from behind an obstruction. The behavior is inherently abrupt and unpredictable, creating a high-risk situation similar to ghost probing.
    Example: A vehicle or object accelerates suddenly and veers into the self-driving vehicle's path at a dangerously close distance.
    
    2.Inability to Anticipate the Object's Movement:
    Description: Due to the high speed or erratic behavior of the moving object, the self-driving vehicle's driver cannot predict its approach in time. This unpredictability creates a scenario similar to ghost probing, where the self-driving vehicle has minimal time to react.
    Example: A vehicle or object changes direction suddenly while speeding and moves dangerously close to the self-driving vehicle.

    Annotation Note: These cases align with the **emergency** and **unpredictable nature** of ghost probing, as they share the same high-risk factors: **minimal reaction time, sudden appearance in the vehicle's path, and immediate danger of collision**.

2) Cut-In(加塞):
    Definition:  Typically **within same-direction traffic flow**, a cut-in happens when a vehicle deliberately forces its way in front of another vehicle's traffic lane from the **adjacent lane**, occupying another driver's lane space.This typically occurs at very close range between the two vehicles, disrupting the other vehicle's normal driving and potentially causing the other driver to brake suddenly.
    加塞是指在**同向**车流行驶过程中，某车辆从**侧面相邻车道**强行插入其他车辆的行驶路线,强行抢占他人车道的行驶空间，这种情况下一般是指距离非常近，从而影响其他车辆的正常行驶，甚至导致紧急刹车。
    Characteristics:
    A cut-in is defined only when a vehicle merges into the current lane from an adjacent side lane.
    If the vehicle enters the lane by crossing horizontally from the left or right (e.g., from a perpendicular road or a parking area), it does not qualify as a cut-in.
    Cut-in特点: 只有从相邻车道侧面插入进当前车道才算cut-in, 如果是从左右手两边的垂直的路上横插过来不算cut-in.
    ### Key Rules:
    1. Cut-in occurs ONLY when a vehicle merges from an adjacent side lane.
    2. Entry from perpendicular or non-adjacent lanes is NOT "cut-in."

    ### Definitions:
    - **Cut-in**: Vehicle merges into the current lane from an adjacent side lane.
    - **Non Cut-in**: Vehicle enters the current lane from a perpendicular road or crosses horizontally.

    ### Classification Examples:
    - **正例 (Cut-in)**:
    - A car from the adjacent left lane merges into the self-vehicle's lane abruptly.
    - **反例 (Non Cut-in)**:
    - A car enters from a perpendicular road on the right. "Key Actions": "none."
    注意: 任何来自垂直侧路的插入merge均属于 "side road merging"，而非 cut-in。

    ### Classification Flow:
    1. Adjacent lane merge → "cut-in."
    2. Perpendicular road entry → "none."

    ### Output Format:
    Always output the JSON object with `key_actions` strictly adhering to the classification rules.

    Example 2:
    - A pedestrian crossing from the right appears suddenly in the lane. 
    "key_actions": "ghost probing"
    **注意**: 任何来自垂直侧路的插入均属于 "side road merging"，而非 cut-in。

    ### Cut-in Decision Guide ###
    1. Did the vehicle enter from an **adjacent lane**?
    - Yes: Label as "cut-in."
    - No: Continue to Step 2.

    2. Did the vehicle enter from a **perpendicular side road** or driveway?
    - Yes: Label as "none."
    - No: Continue to Step 3.

    3. Is the vehicle behavior ambiguous or unclear?
    - Yes: Label as "none."
    - No: Use specific action like "overtaking."             

    ***Key Note***
    Vehicles entering from a perpendicular or non-adjacent road should never be labeled as "cut-in". Instead, they may be classified as "none" or "ghost probing" if relevant.           

**Validation Process (Simulated Post-Processing):**
  - After generating key_actions, review each label:
    - If "cut-in" is present but the description involves a perpendicular road or side road, revise key_actions to "none".
    - If unsure, label as "none" to avoid errors.
    如果你不能分辨cut-in, 要尽可能用最小范围的推理. 每当你标识为cut-in, 你要在后面加上reasoning过程.


3) left/right-side Overtaken: refers to another vehicle accelerating from the left or right lane, initially from behind, accelerating, until passing the ego vehicle, and moving ahead to take the lead position.
Here, only labelling legal overtaking from the left or right, which means the vehicle maintains a safe distance, overtakes the ego vehicle by more than one or two car lengths, and does not pose any danger. 
Note: Focus only on other vehicles overtaking the ego vehicle, there is no need to label the ego vehicle overtaking other vehicles.
That's why this labeled behavior is called "left/right-side overtaken" rather than "overtaking", as it is described from the perspective of the ego vehicle. Do not label anything as "left/right-side overtaking".

Your angle appears to watch video frames recorded from a surveillance camera in a car. Your role should focus on detecting and predicting dangerous actions in a "Ghosting" manner
where pedestrians in the scene might suddenly appear in front of the current car. This could happen if the pedestrian suddenly darts out from behind an obstacle in the driver's view.
This behavior is extremly dangerous because it gives the driver very little time to react. 
Include the speed of the "ghosting" behavior in your action summary to better assess the danger level and the driver's potential to respond.

Provide detailed description of the people's behavior and potential dangerous actions that could lead to collisions. Describe how you think the individual could crash into the car, and explain your deduction process. Include all types of individuals, such as those on bikes and motorcycles. 
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

Your goal is to create the best action summary you can. Always and only return valid JSON.
You must always and only answer totally in **English** language!!! Ensure all parts of the JSON output, including **summaries**, **actions**, **next_action**, and **THE WHOLE OUTPUT**, **MUST BE IN ENGLISH** If you answer ANY word in Chinese, you are fired immediately! Translate Chinese to English if there is Chinese in "next_action" field.
"""

        # 替换占位符
        system_prompt = system_prompt.replace("{video_id}", video_id)
        system_prompt = system_prompt.replace("{segment_id_str}", segment_id_str)
                
        # 设置初始的系统提示
        # 注意: Gemini 1.5 Flash API使用safety settings来控制系统行为
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT", 
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT", 
                "threshold": "BLOCK_ONLY_HIGH"
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
                
                # 创建模型实例
                model = genai.GenerativeModel(GEMINI_MODEL)
                
                # 准备请求内容
                contents = []
                
                # 加入系统提示作为第一条消息
                # 最新的API不使用system_instruction，而是将系统提示作为第一条消息
                contents.append(system_prompt)
                
                # 添加文本提示
                for p in prompt_parts:
                    contents.append(p)
                
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
                    contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings
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
    
    # 按间隔处理视频，确保严格按照frame_interval间隔来划分
    while current_interval < video_duration:
        # 计算当前间隔的开始和结束时间
        interval_start = current_interval
        # 严格确保interval_end是10秒的倍数，除非是视频末尾
        if interval_start + frame_interval >= video_duration:
            interval_end = video_duration
        else:
            interval_end = interval_start + frame_interval
        
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
        # 修复current_interval的更新逻辑，确保是严格按frame_interval增加
        # 之前的逻辑是current_interval = interval_end，这可能导致不规则间隔
        if current_interval + frame_interval > video_duration:
            current_interval = video_duration  # 如果到达视频末尾
        else:
            current_interval += frame_interval  # 严格增加frame_interval
        intervals_processed += 1
        progress_bar.update(1)
    
    # 关闭资源
    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()
    
    print('视频提取、分析和转录完成！')
    
    # 生成以视频文件名命名的结果文件
    result_filename = f'actionSummary_{video_filename.split(".")[0]}.json'
    
    # 将完整结果保存到指定的输出目录中
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

def process_all_videos(video_folder, frame_interval=10, frames_per_interval=10, speed_mode=False, start_at=0, limit=None, skip_existing=True, retry_failed=False, output_dir='result/gemini-testinterval'):
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
        result_path = os.path.join('result', result_filename)
        
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
        # 访问全局变量
        final_arr.clear()  # 清空列表而不是重新赋值
        
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
                    
                    # 实际处理视频
                    AnalyzeVideo(video_path, frame_interval, frames_per_interval, speed_mode)
                    
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
        
        overall_progress.update(1)
    
    return successful, failed, skipped, videos_duration 

# 添加主函数
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
    parser.add_argument('--output-dir', type=str, default='result/gemini-testinterval', help='输出目录')
    parser.add_argument('--processes', type=int, default=1, help='并行处理的进程数，默认为1 (注意：Gemini可能对并行请求有限制)')
    parser.add_argument('--speed-mode', action='store_true', help='启用速度优化模式，减少API调用和帧处理以提高速度')
    parser.add_argument('--video-list', type=str, default='', help='包含视频文件名列表的文件路径')
    parser.add_argument('--model', type=str, default='gemini-1.5-flash', help='使用的Gemini模型，默认为gemini-1.5-flash')
    
    args = parser.parse_args()
    
    # 设置Gemini模型
    if args.model:
        # 使用global声明访问上层作用域的变量
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
        # 批量处理视频
        try:
            video_folder = args.folder
            print(f"开始批量处理 {video_folder} 中的视频")
            detailed_logger.info(f"开始批量处理 {video_folder} 中的视频")

            # 获取文件夹中的所有视频文件
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = []
            
            for filename in os.listdir(video_folder):
                file_path = os.path.join(video_folder, filename)
                # 检查是否是文件且扩展名在指定列表中
                if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(file_path)
                    detailed_logger.debug(f"找到视频文件: {file_path}")
            
            # 排序视频文件以确保处理顺序一致
            video_files.sort()
            detailed_logger.debug(f"视频文件已排序")
            
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
                detailed_logger.info(f"由于设置了限制，将只处理前 {args.limit} 个视频")
            
            # 处理每个视频
            total_videos = len(video_files)
            print(f"共找到 {total_videos} 个视频文件，开始处理")
            
            successful = 0
            failed = 0
            skipped = 0
            
            # 创建总体进度条
            overall_progress = tqdm.tqdm(total=total_videos, desc="总体进度", unit="视频")
            
            for i, video_path in enumerate(video_files):
                video_name = os.path.basename(video_path)
                # 显示明显的进度信息
                progress_percentage = ((i+1) / total_videos) * 100
                print(f"\n处理视频 [{i+1}/{total_videos}]: {video_name} ({progress_percentage:.1f}%)")
                
                # 检查是否已经处理过该视频
                result_filename = f'actionSummary_{video_name.split(".")[0]}.json'
                result_path = os.path.join(args.output_dir, result_filename)
                
                if os.path.exists(result_path) and not args.no_skip and not args.retry_failed:
                    print(f"视频 {video_name} 已经处理过，跳过")
                    skipped += 1
                    overall_progress.update(1)
                    continue
                
                # 在处理每个新视频前重置final_arr
                # 访问全局变量
                final_arr.clear()  # 清空列表而不是重新赋值
                
                try:
                    # 处理视频
                    AnalyzeVideo(video_path, args.interval, args.frames, args.speed_mode)
                    print(f"成功处理视频: {video_name}")
                    successful += 1
                except Exception as e:
                    print(f"处理视频失败: {video_name}, 错误: {str(e)}")
                    detailed_logger.error(f"处理视频失败: {video_name}, 错误: {str(e)}")
                    detailed_logger.error(traceback.format_exc())
                    failed += 1
                
                overall_progress.update(1)
            
            # 关闭进度条
            overall_progress.close()
            
            print(f"\n处理完成: 成功 {successful} 个, 失败 {failed} 个, 跳过 {skipped} 个")
            detailed_logger.info(f"处理完成: 成功 {successful} 个, 失败 {failed} 个, 跳过 {skipped} 个")
            
        except Exception as e:
            print(f"批量处理过程中出错: {str(e)}")
            detailed_logger.error(f"批量处理过程中出错: {str(e)}")
            detailed_logger.error(traceback.format_exc()) 