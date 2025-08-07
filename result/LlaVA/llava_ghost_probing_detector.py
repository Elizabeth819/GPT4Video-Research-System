#!/usr/bin/env python3
"""
LLaVA-NeXT Ghost Probing Detection Script
基于LLaVA-Video-7B-Qwen2模型的鬼探头视频打标系统
使用与GPT-4.1相同的平衡提示词以确保评估一致性
文件路径: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/llava_ghost_probing_detector.py
"""

import os
import sys
import json
import logging
import warnings
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import copy

# 导入LLaVA-NeXT相关模块
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LLaVA-NeXT'))
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llava_ghost_probing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLaVAGhostProbingDetector:
    """LLaVA-NeXT鬼探头检测器"""
    
    def __init__(self, 
                 model_name: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
                 max_frames: int = 64,
                 device: str = "cuda"):
        """
        初始化LLaVA鬼探头检测器
        
        Args:
            model_name: LLaVA模型名称
            max_frames: 最大帧数
            device: 计算设备
        """
        self.model_name = model_name
        self.max_frames = max_frames
        self.device = device
        
        logger.info(f"初始化LLaVA Ghost Probing Detector")
        logger.info(f"模型: {model_name}")
        logger.info(f"最大帧数: {max_frames}")
        logger.info(f"设备: {device}")
        
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """加载LLaVA模型"""
        try:
            logger.info("正在加载LLaVA模型...")
            
            # 加载预训练模型
            self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
                self.model_name, 
                None, 
                "llava_qwen", 
                torch_dtype="bfloat16", 
                device_map="auto"
            )
            
            # 设置模型为评估模式
            self.model.eval()
            
            # 对话模板
            self.conv_template = "qwen_1_5"
            
            logger.info("✅ LLaVA模型加载成功")
            
        except Exception as e:
            logger.error(f"❌ LLaVA模型加载失败: {e}")
            raise
    
    def load_video(self, video_path: str, fps: int = 1, force_sample: bool = True) -> Tuple[np.ndarray, str, float]:
        """
        加载视频并提取关键帧
        
        Args:
            video_path: 视频文件路径
            fps: 帧率采样频率
            force_sample: 是否强制采样到max_frames数量
            
        Returns:
            spare_frames: 提取的视频帧
            frame_time: 帧时间字符串
            video_time: 视频总时长
        """
        try:
            if self.max_frames == 0:
                return np.zeros((1, 336, 336, 3)), "0.00s", 0.0
            
            # 使用decord读取视频
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frame_num = len(vr)
            video_time = total_frame_num / vr.get_avg_fps()
            
            # 计算采样间隔
            sample_fps = round(vr.get_avg_fps() / fps)
            frame_idx = [i for i in range(0, len(vr), sample_fps)]
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
            
            # 如果帧数超过max_frames或强制采样，进行均匀采样
            if len(frame_idx) > self.max_frames or force_sample:
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frames, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frame_time = [i / vr.get_avg_fps() for i in frame_idx]
            
            # 格式化时间字符串
            frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
            
            # 提取帧
            spare_frames = vr.get_batch(frame_idx).asnumpy()
            
            logger.info(f"视频加载完成: {len(spare_frames)}帧, 总时长{video_time:.2f}秒")
            
            return spare_frames, frame_time_str, video_time
            
        except Exception as e:
            logger.error(f"视频加载失败 {video_path}: {e}")
            raise
    
    def create_ghost_probing_prompt(self, 
                                  video_id: str, 
                                  frame_time_str: str, 
                                  video_time: float, 
                                  num_frames: int) -> str:
        """
        创建鬼探头检测提示词 - 使用与GPT-4.1相同的平衡提示词
        
        Args:
            video_id: 视频ID
            frame_time_str: 帧时间字符串
            video_time: 视频总时长
            num_frames: 帧数量
            
        Returns:
            格式化的提示词
        """
        
        # 🔧 与GPT-4.1完全相同的平衡提示词
        system_content = f"""You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {video_time:.1f} seconds of audio from a video,
as well as {num_frames} frames split evenly throughout {video_time:.1f} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({num_frames}
frames over {video_time:.1f} seconds), which is generated from your analysis of each frame ({num_frames} in total),
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
    "segment_id": "Segment_001",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "{video_time:.1f}s",
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

The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time_str}. Please answer the following questions related to this video.

Audio Transcription: No audio available for this analysis.
"""
        
        return system_content
    
    def analyze_video(self, video_path: str, video_id: Optional[str] = None) -> Optional[Dict]:
        """
        分析单个视频进行鬼探头检测
        
        Args:
            video_path: 视频文件路径
            video_id: 视频ID，如果为None则从文件名提取
            
        Returns:
            分析结果字典，如果失败返回None
        """
        try:
            # 提取视频ID
            if video_id is None:
                video_id = Path(video_path).stem
            
            logger.info(f"🎬 开始分析视频: {video_id}")
            
            # 1. 加载视频帧
            video_frames, frame_time_str, video_time = self.load_video(video_path)
            
            # 2. 预处理视频帧
            video_tensor = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(self.device).bfloat16()
            video_list = [video_tensor]
            
            # 3. 创建提示词
            prompt_text = self.create_ghost_probing_prompt(
                video_id, frame_time_str, video_time, len(video_frames)
            )
            
            # 4. 构建对话
            question = DEFAULT_IMAGE_TOKEN + f"{prompt_text}\nPlease analyze this video for ghost probing detection."
            conv = copy.deepcopy(conv_templates[self.conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            
            # 5. Token化
            input_ids = tokenizer_image_token(
                prompt_question, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors="pt"
            ).unsqueeze(0).to(self.device)
            
            # 6. 生成回复
            logger.info("🧠 正在进行LLaVA推理...")
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    images=video_list,
                    modalities=["video"],
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=4096,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 7. 解码输出
            text_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
            
            # 8. 提取JSON结果
            json_result = self._extract_json_from_response(text_output)
            
            if json_result:
                logger.info(f"✅ 视频分析成功: {video_id}")
                return json_result
            else:
                logger.error(f"❌ 无法从响应中提取有效JSON: {video_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 视频分析失败 {video_path}: {e}")
            return None
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """
        从LLaVA响应中提取JSON结果
        
        Args:
            response: LLaVA原始响应
            
        Returns:
            解析的JSON字典，如果失败返回None
        """
        try:
            # 查找JSON开始和结束位置
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx+1]
                
                # 尝试解析JSON
                result = json.loads(json_str)
                return result
            else:
                logger.warning("响应中未找到完整的JSON结构")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.debug(f"原始响应: {response}")
            return None
        except Exception as e:
            logger.error(f"JSON提取失败: {e}")
            return None
    
    def extract_ghost_probing_label(self, analysis_result: Dict) -> Tuple[str, float]:
        """
        从分析结果中提取鬼探头标签
        
        Args:
            analysis_result: LLaVA分析结果
            
        Returns:
            (标签, 置信度) - 标签为"ghost_probing", "potential_ghost_probing", 或"normal"
        """
        try:
            key_actions = analysis_result.get("key_actions", "").lower()
            
            # 根据关键动作判断类别
            if "ghost probing" in key_actions and "potential" not in key_actions:
                return "ghost_probing", 0.9
            elif "potential ghost probing" in key_actions:
                return "potential_ghost_probing", 0.7
            else:
                return "normal", 0.8
                
        except Exception as e:
            logger.error(f"标签提取失败: {e}")
            return "normal", 0.5

def main():
    """测试主函数"""
    # 测试单个视频
    detector = LLaVAGhostProbingDetector()
    
    # 测试视频路径
    test_video = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/images_1_001.avi"
    
    if os.path.exists(test_video):
        result = detector.analyze_video(test_video)
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 提取鬼探头标签
            label, confidence = detector.extract_ghost_probing_label(result)
            print(f"\n鬼探头检测结果: {label} (置信度: {confidence})")
        else:
            print("❌ 视频分析失败")
    else:
        print(f"❌ 测试视频不存在: {test_video}")

if __name__ == "__main__":
    main()