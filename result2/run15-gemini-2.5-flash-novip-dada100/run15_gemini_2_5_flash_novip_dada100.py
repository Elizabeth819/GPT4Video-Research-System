#!/usr/bin/env python3
"""
Run 15: Gemini 2.5 Flash + Simple Prompt (No Few-shot) - 100 Videos
基于Run 8的Paper Batch prompt，测试Gemini 2.5 Flash在简单prompt下的性能
对比实验：Run 15 (无Few-shot) vs Run 16 (有Few-shot)
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

class Run15GeminiAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.setup_gemini()
        self.setup_directories()
        
    def setup_logging(self):
        """设置日志记录"""
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"run15_gemini_2_5_flash_{self.timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Run 15 Gemini 2.5 Flash + Simple Prompt Analysis Started - {self.timestamp}")
        
    def setup_gemini(self):
        """设置Gemini API - 支持双密钥策略"""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.api_key_2 = os.environ.get("GEMINI_API_KEY_2")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.current_api_key = self.api_key
        self.logger.info(f"Gemini 2.5 Flash model initialized: {self.api_key[:10]}...")
        if self.api_key_2:
            self.logger.info(f"Backup API key available: {self.api_key_2[:10]}...")
        
    def switch_api_key(self):
        """切换到备用API密钥"""
        if self.api_key_2 and self.current_api_key == self.api_key:
            self.current_api_key = self.api_key_2
            genai.configure(api_key=self.api_key_2)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
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
        
    def get_simple_prompt(self, video_id, frame_interval=10, frames_per_interval=10):
        """获取简单版本的Paper Batch prompt（无Few-shot Examples）"""
        system_content = f"""You are an expert driver assistance AI system specialized in analyzing driving scenarios for autonomous vehicles. 
Your primary task is to detect and classify dangerous driving behaviors, particularly focusing on "ghost probing" incidents.

The Current Action Summary is built by doing the following, given a series of frames 
(extracted at {frame_interval} seconds, containing {frames_per_interval} 
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the portion of the video you are considering.

Direction - Please identify the objects in the image based on their position in the image itself. Do not assume your own position within the image. Treat the left side of the image as 'left' and the right side of the image as 'right'. Assume the viewpoint is standing from at the bottom center of the image. Describe whether the objects are on the left or right side of this central point, left is left, and right is right. For example, if there is a car on the left side of the image, state that the car is on the left.

**Task 1: Identify and Predict potential "Ghost Probing(专业术语：鬼探头)" behavior**

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

Note: Only those emerging from behind a physical obstruction can be considered as 鬼探头 (ghost probing).

**Task 2: Explain Current Driving Actions**
Analyze the current video frames to extract actions. Describe not only the actions themselves but also provide detailed reasoning for why the vehicle is taking these actions, such as changes in speed and direction. Focus solely on the reasoning for the vehicle's actions, excluding any descriptions of pedestrian behavior. Explain why the driver is driving at a certain speed, making turns, or stopping. Your goal is to provide a comprehensive understanding of the vehicle's behavior based on the visual data. Output in the "actions" field of the JSON format template.

**Task 3: Predict Next Driving Action**
Understand the current road conditions, the driving behavior, and to predict the next driving action. Analyze the video and audio to provide a comprehensive summary of the road conditions, including weather, traffic density, road obstacles, and traffic light if visible. Predict the next driving action based on two dimensions, one is driving speed control, such as accelerating, braking, turning, or stopping, the other one is to predict the next lane control, such as change to left lane, change to right lane, keep left in this lane, keep right in this lane, keep straight. Your summary should help understand not only what is happening at the moment but also what is likely to happen next with logical reasoning. The principle is safety first, so the prediction action should prioritize the driver's safety and secondly the pedestrians' safety. Be incredibly detailed. Output in the "next_action" field of the JSON format template.

**Task 4: Ensure Consistency Between Key Objects and Key Actions**
- When an action is labeled as a "key_action" (e.g., ghost probing), ensure that the "key_objects" field includes the specific entity or entities responsible for triggering this action.

Additional Requirements:
- `key_actions` must strictly adhere to the predefined categories:
    - ghost probing
    - overtaking, specify "left-side overtaking" or "right-side overtaking" when relevant.
    - none (if no dangerous behavior is observed)

- All textual fields must be in English.
- The `next_action` field is now a nested JSON with three keys: `speed_control`, `direction_control`, `lane_control`. Each must choose one value from their respective sets.

Your response should be a valid JSON object with the following EXACT structure:
{{
    "video_id": "{video_id}",
    "segment_id": "full_video",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "{frame_interval}.0s",
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

Audio Transcription: [No audio available for this analysis]
"""
        return system_content
        
    def extract_frames_from_video(self, video_path, interval=10, max_frames=10):
        """从视频中提取帧 - 基于Paper Batch脚本的参数"""
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
        """使用Gemini 2.5 Flash分析视频 - Simple Prompt版本"""
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
            system_prompt = self.get_simple_prompt(video_id, frame_interval=10, frames_per_interval=10)
            
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
                    'run_id': 'Run 15',
                    'timestamp': self.timestamp,
                    'model': 'gemini-2.5-flash',
                    'prompt_version': 'Simple Paper Batch (No Few-shot)',
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
                        'Simple Paper Batch Prompt',
                        'No Few-shot Examples',
                        'Dual API Key Strategy',
                        'Enhanced Error Handling',
                        'Temperature=0 for Consistency'
                    ]
                },
                'results': results
            }
            
            summary_file = self.output_dir / f"run15_analysis_summary_{self.timestamp}.json"
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
    print("🚀 Run 15: Gemini 2.5 Flash + Simple Prompt (No Few-shot)")
    print("=" * 60)
    
    try:
        analyzer = Run15GeminiAnalyzer()
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