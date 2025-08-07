#!/usr/bin/env python3
"""
GPT-4.1平衡版100视频鬼探头检测
处理images_1_001到images_5_xxx的所有100个视频
使用与GPT-4.1相同的prompt和格式
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import time
import hashlib
from typing import Dict, List, Optional
import requests
import base64
from PIL import Image
import cv2
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPT41BalancedDetector:
    """GPT-4.1平衡版鬼探头检测器"""
    
    def __init__(self):
        """初始化检测器"""
        # Azure OpenAI配置 (使用环境变量)
        self.api_key = os.getenv('AZURE_VISION_KEY', 'placeholder-key')
        self.endpoint = os.getenv('VISION_ENDPOINT', 'https://placeholder.openai.azure.com/')
        self.deployment_name = os.getenv('VISION_DEPLOYMENT_NAME', 'gpt-4o-vision')
        self.api_version = "2024-02-15-preview"
        
        # 配置参数 (与GPT-4.1保持一致)
        self.frame_interval = 10  # 每段10秒
        self.frames_per_interval = 10  # 每段10帧
        self.max_tokens = 2000
        self.temperature = 0
        self.max_retry_attempts = 3
        
        self.processed_videos = []
        
        logger.info("🔧 GPT-4.1平衡版检测器初始化完成")
        logger.info(f"📊 配置: {self.frames_per_interval}帧/{self.frame_interval}秒, 温度={self.temperature}")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图像编码为base64"""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ 图像编码失败 {image_path}: {e}")
            return ""
    
    def extract_video_frames(self, video_path: str) -> List[str]:
        """提取视频关键帧"""
        logger.info(f"🎬 提取视频帧: {Path(video_path).name}")
        
        try:
            # 创建临时目录
            temp_dir = Path("./temp_frames")
            temp_dir.mkdir(exist_ok=True)
            
            # 使用OpenCV读取视频
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"📊 视频信息: {total_frames}帧, {fps:.2f}fps, {duration:.2f}秒")
            
            # 计算需要提取的帧索引
            if duration <= self.frame_interval:
                # 视频短于间隔，提取所有关键帧
                frame_indices = np.linspace(0, total_frames - 1, min(self.frames_per_interval, total_frames), dtype=int)
            else:
                # 视频较长，提取前frame_interval秒的帧
                target_frames = int(fps * self.frame_interval)
                frame_indices = np.linspace(0, min(target_frames - 1, total_frames - 1), self.frames_per_interval, dtype=int)
            
            # 提取帧
            frame_paths = []
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_path = temp_dir / f"frame_{i:03d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                    
                    # 验证帧质量
                    if frame.mean() < 10:  # 检查是否为黑帧
                        logger.warning(f"⚠️  帧 {i} 可能为黑帧")
            
            cap.release()
            logger.info(f"✅ 成功提取 {len(frame_paths)} 帧")
            
            return frame_paths
            
        except Exception as e:
            logger.error(f"❌ 视频帧提取失败: {e}")
            return []
    
    def create_balanced_gpt41_prompt(self, video_id: str, trans: str = "无音频") -> str:
        """创建平衡版GPT-4.1 prompt"""
        
        system_content = f"""You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {self.frame_interval} seconds of audio from a video,
as well as {self.frames_per_interval} frames split evenly throughout {self.frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({self.frames_per_interval}
frames over {self.frame_interval} seconds), which is generated from your analysis of each frame ({self.frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the video.

IMPORTANT: For ghost probing detection, consider TWO categories:

**1. HIGH-CONFIDENCE Ghost Probing (use "ghost probing" in key_actions)**:
- Object appears EXTREMELY close (within 1-2 vehicle lengths, <3 meters) 
- Appearance is SUDDEN and from blind spots (behind parked cars, buildings, corners)
- Occurs in HIGH-RISK environments: highways, rural roads, parking lots, uncontrolled intersections
- Requires IMMEDIATE emergency braking/swerving to avoid collision
- Movement is COMPLETELY UNPREDICTABLE and violates traffic expectations

**2. POTENTIAL Ghost Probing (use "potential ghost probing" in key_actions)**:
- Object appears suddenly but at moderate distance (3-5 meters)
- Sudden movement in environments where some unpredictability exists
- Requires emergency braking but collision risk is moderate
- Movement is unexpected but not completely impossible given the context

**3. NORMAL Traffic Situations (do NOT use "ghost probing")**:
- Pedestrians crossing at intersections, crosswalks, or traffic lights
- Vehicles making normal lane changes, turns, or merging with signals
- Cyclists following predictable paths in urban areas or bike lanes
- Any movement that is EXPECTED given the traffic environment and context

**Environment Context Guidelines**:
- INTERSECTION/CROSSWALK: Expect pedestrians and cyclists - use "emergency braking due to pedestrian crossing"
- HIGHWAY/RURAL: Higher chance of genuine ghost probing - be more sensitive
- PARKING LOT: Expect sudden vehicle movements - use "potential ghost probing" if very sudden
- URBAN STREET: Mixed - consider visibility and predictability

Use "ghost probing" for clear cases, "potential ghost probing" for borderline cases, and descriptive terms for normal traffic situations.

Your response should be a valid JSON object with the following EXACT structure (match this format precisely):
{{
    "video_id": "{video_id}",
    "segment_id": "segment_1",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "{self.frame_interval}.0s",
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

Audio Transcription: {trans}"""

        return system_content
    
    def call_gpt41_vision_api(self, system_prompt: str, frame_paths: List[str]) -> Optional[Dict]:
        """调用GPT-4.1 Vision API"""
        
        try:
            # 准备图像数据
            images_data = []
            for frame_path in frame_paths:
                base64_image = self.encode_image_to_base64(frame_path)
                if base64_image:
                    images_data.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            
            if not images_data:
                logger.error("❌ 没有有效的图像数据")
                return None
            
            # 构建请求
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            # 消息格式
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": images_data
                }
            ]
            
            payload = {
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            
            # API调用
            url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
            
            logger.info(f"🔍 调用GPT-4.1 API: {len(images_data)}张图像")
            start_time = time.time()
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            api_time = time.time() - start_time
            logger.info(f"⏱️  API调用时间: {api_time:.2f}秒")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # 尝试解析JSON
                try:
                    # 清理可能的markdown格式
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        content = content.split('```')[1].strip()
                    
                    parsed_result = json.loads(content)
                    logger.info("✅ GPT-4.1 分析完成")
                    return parsed_result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON解析失败: {e}")
                    logger.error(f"原始响应: {content[:500]}...")
                    return None
            else:
                logger.error(f"❌ API调用失败: {response.status_code}")
                logger.error(f"错误信息: {response.text[:500]}...")
                return None
                
        except Exception as e:
            logger.error(f"❌ GPT-4.1 API调用异常: {e}")
            return None
    
    def process_single_video(self, video_path: str) -> Optional[Dict]:
        """处理单个视频"""
        video_name = Path(video_path).stem
        logger.info(f"🎬 开始处理视频: {video_name}")
        
        start_time = time.time()
        
        try:
            # 1. 提取视频帧
            frame_paths = self.extract_video_frames(video_path)
            if not frame_paths:
                logger.error(f"❌ 无法提取视频帧: {video_name}")
                return None
            
            # 2. 创建prompt
            system_prompt = self.create_balanced_gpt41_prompt(video_name)
            
            # 3. 调用GPT-4.1 API
            result = None
            for attempt in range(self.max_retry_attempts):
                logger.info(f"🔄 尝试 {attempt + 1}/{self.max_retry_attempts}")
                result = self.call_gpt41_vision_api(system_prompt, frame_paths)
                
                if result:
                    break
                    
                if attempt < self.max_retry_attempts - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
            
            # 4. 清理临时文件
            for frame_path in frame_paths:
                try:
                    Path(frame_path).unlink()
                except:
                    pass
            
            processing_time = time.time() - start_time
            
            if result:
                # 添加处理元数据
                result.update({
                    'processing_time': round(processing_time, 2),
                    'model': 'GPT-4.1-Balanced',
                    'timestamp': datetime.now().isoformat(),
                    'frames_analyzed': len(frame_paths),
                    'api_config': {
                        'frame_interval': self.frame_interval,
                        'frames_per_interval': self.frames_per_interval,
                        'temperature': self.temperature,
                        'max_tokens': self.max_tokens
                    }
                })
                
                logger.info(f"✅ 处理完成: {video_name} ({processing_time:.2f}s)")
                return result
            else:
                logger.error(f"❌ 处理失败: {video_name}")
                return None
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 处理异常: {video_name} - {e} ({processing_time:.2f}s)")
            return None
    
    def process_100_videos(self, video_folder: str) -> List[Dict]:
        """处理100个视频"""
        
        video_folder_path = Path(video_folder)
        if not video_folder_path.exists():
            logger.error(f"❌ 视频文件夹不存在: {video_folder}")
            return []
        
        # 查找所有视频文件 (images_1_001 到 images_5_xxx)
        video_files = []
        for pattern in ["images_1_*.avi", "images_2_*.avi", "images_3_*.avi", "images_4_*.avi", "images_5_*.avi"]:
            video_files.extend(list(video_folder_path.glob(pattern)))
        
        video_files.sort()  # 确保顺序
        
        if not video_files:
            logger.error(f"❌ 未找到视频文件: {video_folder}")
            return []
        
        logger.info(f"📊 找到 {len(video_files)} 个视频文件")
        logger.info(f"📊 范围: {video_files[0].name} 到 {video_files[-1].name}")
        
        # 处理所有视频
        results = []
        failed_count = 0
        
        print("=" * 80)
        print("🚀 GPT-4.1平衡版100视频鬼探头检测")
        print("=" * 80)
        print(f"📊 总视频数: {len(video_files)}")
        print(f"🎯 使用模型: GPT-4.1 Balanced Prompt")
        print(f"⚙️  配置: {self.frames_per_interval}帧/{self.frame_interval}秒")
        print("=" * 80)
        
        for i, video_file in enumerate(video_files):
            print(f"\n📹 处理视频 {i+1}/{len(video_files)}: {video_file.name}")
            
            result = self.process_single_video(str(video_file))
            
            if result:
                results.append(result)
                
                # 提取关键信息
                ghost_probing = "ghost probing" in result.get('key_actions', '').lower()
                potential_ghost = "potential ghost probing" in result.get('key_actions', '').lower()
                
                if ghost_probing:
                    print(f"🚨 高置信度鬼探头检测")
                elif potential_ghost:
                    print(f"⚠️  潜在鬼探头检测") 
                else:
                    print(f"✅ 正常交通场景")
                    
                print(f"📊 处理时间: {result.get('processing_time', 0):.2f}s")
                
            else:
                failed_count += 1
                print(f"❌ 处理失败")
                
                # 创建失败记录
                results.append({
                    'video_id': video_file.stem,
                    'error': 'Processing failed',
                    'timestamp': datetime.now().isoformat()
                })
            
            # 每10个视频保存一次中间结果
            if (i + 1) % 10 == 0:
                self.save_intermediate_results(results, i + 1)
        
        print("\n" + "=" * 80)
        print("🎉 100视频处理完成!")
        print("=" * 80)
        print(f"✅ 成功处理: {len(results) - failed_count}")
        print(f"❌ 处理失败: {failed_count}")
        print(f"📊 成功率: {((len(results) - failed_count) / len(video_files) * 100):.1f}%")
        
        return results
    
    def save_intermediate_results(self, results: List[Dict], count: int):
        """保存中间结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"./outputs/results/gpt41_balanced_intermediate_{count}_{timestamp}.json"
            
            os.makedirs("./outputs/results", exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'model': 'GPT-4.1-Balanced',
                        'processed_count': count,
                        'timestamp': timestamp,
                        'config': {
                            'frame_interval': self.frame_interval,
                            'frames_per_interval': self.frames_per_interval,
                            'temperature': self.temperature
                        }
                    },
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 中间结果已保存: {filename}")
            
        except Exception as e:
            logger.error(f"❌ 保存中间结果失败: {e}")

def main():
    """主函数"""
    
    print("🚀 GPT-4.1平衡版100视频鬼探头检测")
    print("=" * 60)
    
    # 检查环境变量
    required_env_vars = ['AZURE_VISION_KEY', 'VISION_ENDPOINT', 'VISION_DEPLOYMENT_NAME']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ 缺少环境变量: {missing_vars}")
        print("请设置必要的Azure OpenAI配置")
        return
    
    # 获取视频数据路径
    azureml_data_path = os.environ.get('AZUREML_DATAREFERENCE_video_data')
    
    video_folder = None
    if azureml_data_path:
        video_folder = azureml_data_path
        print(f"🔧 从环境变量找到数据路径: {azureml_data_path}")
    else:
        # 本地测试路径
        video_folder = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos"
        if not Path(video_folder).exists():
            print(f"❌ 视频文件夹不存在: {video_folder}")
            return
    
    # 创建输出目录
    os.makedirs("./outputs/results", exist_ok=True)
    
    # 初始化检测器
    detector = GPT41BalancedDetector()
    
    # 处理100个视频
    results = detector.process_100_videos(video_folder)
    
    if not results:
        print("❌ 未能处理任何视频")
        return
    
    # 保存最终结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 统计结果
    successful_results = [r for r in results if 'error' not in r]
    ghost_probing_count = sum(1 for r in successful_results 
                             if 'ghost probing' in r.get('key_actions', '').lower())
    potential_ghost_count = sum(1 for r in successful_results 
                               if 'potential ghost probing' in r.get('key_actions', '').lower() 
                               and 'ghost probing' not in r.get('key_actions', '').lower())
    
    final_result = {
        'metadata': {
            'model': 'GPT-4.1-Balanced',
            'prompt_version': 'balanced_final',
            'total_videos': len(results),
            'successful_videos': len(successful_results),
            'failed_videos': len(results) - len(successful_results),
            'ghost_probing_detected': ghost_probing_count,
            'potential_ghost_probing_detected': potential_ghost_count,
            'normal_traffic': len(successful_results) - ghost_probing_count - potential_ghost_count,
            'timestamp': timestamp,
            'config': {
                'frame_interval': detector.frame_interval,
                'frames_per_interval': detector.frames_per_interval,
                'temperature': detector.temperature,
                'max_tokens': detector.max_tokens
            }
        },
        'results': results
    }
    
    # 保存最终结果
    json_file = f"./outputs/results/gpt41_balanced_100_videos_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("🎉 最终结果统计:")
    print("=" * 80)
    print(f"📊 总视频数: {len(results)}")
    print(f"✅ 成功处理: {len(successful_results)}")
    print(f"🚨 高置信度鬼探头: {ghost_probing_count}")
    print(f"⚠️  潜在鬼探头: {potential_ghost_count}")
    print(f"🚗 正常交通: {len(successful_results) - ghost_probing_count - potential_ghost_count}")
    print(f"❌ 处理失败: {len(results) - len(successful_results)}")
    print(f"📄 结果文件: {json_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()