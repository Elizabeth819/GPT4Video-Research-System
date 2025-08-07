#!/usr/bin/env python3
"""
重新处理Run 16失败的视频
"""

import os
import sys
import json
import logging
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import cv2
import tempfile
from tqdm import tqdm
import base64
from PIL import Image
import io
from moviepy.editor import VideoFileClip

# 加载环境变量
load_dotenv()

class FailedVideoRetryRun16:
    def __init__(self):
        self.setup_logging()
        self.setup_gemini()
        self.failed_videos = [
            "images_1_001", "images_1_017", "images_1_025", 
            "images_5_012", "images_5_026"
        ]
        self.video_dir = Path("/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos")
        self.output_dir = Path(__file__).parent
        
    def setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run16-gemini-2.5-flash-fewshot-dada100/retry_run16.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_gemini(self):
        """设置Gemini API"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.logger.info(f"Gemini 2.5 Flash model initialized: {api_key[:10]}...")
        
    def get_simple_prompt_with_fewshot(self):
        """获取带few-shot的简单提示词"""
        return """Analyze this autonomous driving video and identify any dangerous "ghost probing" maneuvers.

Ghost probing refers to: A vehicle suddenly appearing from a blind spot (like behind a large vehicle, building, or roadside obstruction) and aggressively cutting into traffic, creating a dangerous situation.

Here are some examples:

Example 1 - Ghost Probing Detected:
{
  "timestamp": "5-8s",
  "summary": "A white sedan suddenly emerges from behind a large truck and cuts into the left lane aggressively",
  "actions": "vehicle cutting in from blind spot, other cars braking",
  "characters": "white sedan, large truck, multiple vehicles in traffic",
  "key_objects": "truck creating blind spot, traffic lanes, road barriers",
  "key_actions": "ghost probing - white sedan appearing from truck's blind spot and cutting into traffic dangerously",
  "next_action": "traffic adjusting to accommodate sudden lane change"
}

Example 2 - Ghost Probing Detected:
{
  "timestamp": "2-4s", 
  "summary": "A dark SUV emerges from behind roadside construction and merges aggressively into moving traffic",
  "actions": "SUV emerging from obstruction, aggressive merging maneuver",
  "characters": "dark SUV, construction vehicles, other traffic",
  "key_objects": "construction zone, roadside barriers, traffic cones",
  "key_actions": "ghost probing - SUV suddenly appearing from construction blind spot and cutting into traffic",
  "next_action": "nearby vehicles taking evasive action"
}

Example 3 - No Ghost Probing:
{
  "timestamp": "0-10s",
  "summary": "Normal highway driving with vehicles maintaining lanes and following traffic patterns",
  "actions": "steady highway driving, lane keeping, normal following distances",
  "characters": "multiple vehicles in different lanes",
  "key_objects": "highway lanes, road markings, traffic",
  "key_actions": "normal driving behavior, no sudden appearances or aggressive maneuvers",
  "next_action": "continued normal traffic flow"
}

Please analyze the video frames and provide the following information in JSON format:

{
  "timestamp": "time_range_of_action",
  "summary": "brief_description_of_the_scene",
  "actions": "current_actions_happening",
  "characters": "people_or_vehicles_present", 
  "key_objects": "important_objects_in_scene",
  "key_actions": "significant_actions_observed (mention 'ghost probing' if detected)",
  "next_action": "predicted_next_action"
}

Focus specifically on identifying ghost probing behavior. If you observe a vehicle suddenly appearing from a blind spot and cutting into traffic dangerously, clearly mention "ghost probing" in the key_actions field."""

    def extract_frames_from_video(self, video_path, interval=10, max_frames=10):
        """从视频中提取帧 - 基于Run 16原始方法"""
        try:
            frames = []
            video_clip = VideoFileClip(str(video_path))
            duration = video_clip.duration
            
            # 计算帧提取时间点
            for i in range(0, int(duration), interval):
                end_time = min(i + interval, duration)
                segment_duration = end_time - i
                
                # 在当前区间内均匀提取帧
                frames_in_segment = min(max_frames, max(1, int(segment_duration)))
                time_step = segment_duration / frames_in_segment if frames_in_segment > 1 else 0
                
                for j in range(frames_in_segment):
                    frame_time = i + (j * time_step)
                    if frame_time < duration:
                        try:
                            frame = video_clip.get_frame(frame_time)
                            
                            # 转换为PIL Image然后调整大小
                            pil_image = Image.fromarray(frame)
                            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                            
                            # 转换为base64
                            buffer = io.BytesIO()
                            pil_image.save(buffer, format='JPEG', quality=85)
                            img_base64 = base64.b64encode(buffer.getvalue()).decode()
                            
                            frames.append({
                                'mime_type': 'image/jpeg',
                                'data': img_base64
                            })
                            
                        except Exception as e:
                            self.logger.warning(f"Error extracting frame at {frame_time}s: {e}")
                            continue
            
            video_clip.close()
            return frames
            
        except Exception as e:
            self.logger.error(f"Error processing video frames: {e}")
            return None
    
    def analyze_with_gemini(self, video_id, max_retries=3):
        """使用Gemini分析视频"""
        video_path = self.video_dir / f"{video_id}.avi"
        
        if not video_path.exists():
            self.logger.error(f"Video file not found: {video_path}")
            return None
            
        self.logger.info(f"Analyzing video: {video_id}")
        
        # 处理视频帧
        frame_images = self.extract_frames_from_video(video_path)
        if not frame_images:
            self.logger.error(f"Failed to process frames for {video_id}")
            return None
        
        prompt = self.get_simple_prompt_with_fewshot()
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"API call attempt {attempt + 1}/{max_retries} for {video_id}")
                
                # 准备内容
                content = [prompt]
                for frame_img in frame_images:
                    content.append({
                        'mime_type': frame_img['mime_type'],
                        'data': frame_img['data']
                    })
                
                # 调用Gemini API
                response = self.model.generate_content(
                    content,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0,
                        max_output_tokens=2048,
                    )
                )
                
                if not response.text:
                    self.logger.error(f"Empty response for {video_id}")
                    continue
                
                # 解析JSON响应
                try:
                    response_text = response.text.strip()
                    
                    # 处理被```json```包装的响应
                    if response_text.startswith('```json'):
                        # 移除```json和```标记
                        response_text = response_text[7:]  # 移除```json
                        if response_text.endswith('```'):
                            response_text = response_text[:-3]  # 移除```
                        response_text = response_text.strip()
                    
                    result = json.loads(response_text)
                    self.logger.info(f"Successfully analyzed {video_id}")
                    return result
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON response for {video_id}: {e}")
                    self.logger.error(f"Response text: {response.text[:500]}...")
                    continue
                    
            except Exception as e:
                self.logger.error(f"API call failed for {video_id} (attempt {attempt + 1}): {e}")
                continue
        
        self.logger.error(f"Failed to analyze {video_id} after {max_retries} attempts")
        return None
    
    def save_result(self, video_id, result):
        """保存分析结果"""
        output_file = self.output_dir / f"actionSummary_{video_id}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved result for {video_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save result for {video_id}: {e}")
            return False
    
    def retry_failed_videos(self):
        """重新处理失败的视频"""
        self.logger.info(f"Starting retry for {len(self.failed_videos)} failed videos")
        
        success_count = 0
        failed_count = 0
        
        for video_id in tqdm(self.failed_videos, desc="Retrying failed videos"):
            try:
                result = self.analyze_with_gemini(video_id)
                if result:
                    if self.save_result(video_id, result):
                        success_count += 1
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                self.logger.error(f"Unexpected error processing {video_id}: {e}")
                failed_count += 1
        
        self.logger.info(f"Retry completed! Success: {success_count}, Failed: {failed_count}")
        return success_count, failed_count

def main():
    retrier = FailedVideoRetryRun16()
    success, failed = retrier.retry_failed_videos()
    
    print(f"\n✅ Retry Results:")
    print(f"📊 Successful: {success}")
    print(f"❌ Failed: {failed}")

if __name__ == "__main__":
    main()