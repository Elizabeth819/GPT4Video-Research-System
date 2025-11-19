#!/usr/bin/env python3
"""
Video-LLaMA2 Ghost Probing Detection System
使用Video-LLaMA2模型进行鬼探头检测分析
针对DADA-2000数据集的100个视频进行分析
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import cv2
from PIL import Image
import decord
from tqdm import tqdm

# 添加Video-LLaMA路径
sys.path.append('/Users/wanmeng/repository/GPT4Video-cobra-auto/result/Video-LLaMA')

# Video-LLaMA2导入
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation, conv_llava_llama_2
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

# 设置decord后端
decord.bridge.set_bridge('torch')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_llama2_ghost_probing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoLLaMA2GhostProbingDetector:
    def __init__(self, 
                 config_path: str = "eval_configs/video_llama_eval_withaudio.yaml",
                 model_type: str = "llama_v2",
                 gpu_id: int = 0,
                 device: str = "cuda:0"):
        """
        初始化Video-LLaMA2鬼探头检测器
        
        Args:
            config_path: 配置文件路径
            model_type: 模型类型 (llama_v2 或 vicuna)
            gpu_id: GPU ID
            device: 设备类型
        """
        self.config_path = config_path
        self.model_type = model_type
        self.gpu_id = gpu_id
        self.device = device
        
        # 初始化模型
        self.model = None
        self.chat = None
        self.vis_processor = None
        
        # 鬼探头检测prompt
        self.ghost_probing_prompt = self._create_ghost_probing_prompt()
        
        logger.info(f"Video-LLaMA2 Ghost Probing Detector initialized")
        logger.info(f"Config: {config_path}")
        logger.info(f"Model Type: {model_type}")
        logger.info(f"Device: {device}")
    
    def _create_ghost_probing_prompt(self) -> str:
        """创建鬼探头检测专用prompt"""
        prompt = """You are VideoAnalyzerGPT analyzing a traffic video for dangerous driving scenarios, specifically "ghost probing" situations.

Your job is to analyze this video and provide a detailed segment-by-segment analysis in the exact JSON format specified below.

For ghost probing detection, consider:
- Objects that suddenly appear very close to the observer vehicle (< 3 meters)
- Minimal warning time for the driver
- Requires immediate emergency response
- Objects emerge from blind spots (behind parked cars, buildings, etc.)
- Situation is unexpected given the traffic context

IMPORTANT: For ghost probing detection, use "ghost probing" in key_actions field when:
- Object appears EXTREMELY close (within 1-2 vehicle lengths, <3 meters) 
- Appearance is SUDDEN and from blind spots
- Requires IMMEDIATE emergency braking/swerving
- Movement is COMPLETELY UNPREDICTABLE

For normal traffic situations, use descriptive terms like:
- "normal intersection start"
- "emergency braking due to pedestrian crossing"
- "maintain safe following distance"

Please respond with a JSON array containing segments of approximately 10-second intervals. Each segment should follow this EXACT format:

[
  {
    "video_id": "video_filename",
    "segment_id": "segment_000",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "10.0s",
    "sentiment": "Positive/Negative/Neutral",
    "scene_theme": "Dramatic/Routine/Dangerous/Safe",
    "characters": "brief description of people in the scene",
    "summary": "comprehensive summary of what happens in this segment",
    "actions": "actions taken by the vehicle and driver responses",
    "key_objects": "numbered list: 1) Position: object description, distance, behavior impact 2) Position: object description, distance, behavior impact",
    "key_actions": "brief description of most important actions (use 'ghost probing' if applicable)",
    "next_action": {
      "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
      "direction_control": "keep direction/turn left/turn right",
      "lane_control": "maintain current lane/change left/change right"
    }
  }
]

Analyze the video carefully and provide the complete segment analysis."""
        
        return prompt
    
    def initialize_model(self):
        """初始化Video-LLaMA2模型"""
        try:
            logger.info("🔄 Initializing Video-LLaMA2 model...")
            
            # 解析配置
            class Args:
                def __init__(self, config_path, model_type, gpu_id):
                    self.cfg_path = config_path
                    self.model_type = model_type
                    self.gpu_id = gpu_id
                    self.options = []
            
            args = Args(self.config_path, self.model_type, self.gpu_id)
            cfg = Config(args)
            
            # 设置随机种子
            seed = cfg.run_cfg.seed + get_rank()
            torch.manual_seed(seed)
            np.random.seed(seed)
            cudnn.benchmark = False
            cudnn.deterministic = True
            
            # 初始化模型
            model_config = cfg.model_cfg
            model_config.device_8bit = args.gpu_id
            model_cls = registry.get_model_class(model_config.arch)
            self.model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
            self.model.eval()
            
            # 初始化视觉处理器
            vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
            self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
            
            # 初始化Chat
            self.chat = Chat(self.model, self.vis_processor, device=f'cuda:{args.gpu_id}')
            
            logger.info("✅ Video-LLaMA2 model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Video-LLaMA2 model: {e}")
            return False
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        分析单个视频的鬼探头情况
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            分析结果字典
        """
        try:
            logger.info(f"📹 Analyzing video: {video_path}")
            
            # 检查文件是否存在
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # 创建对话状态
            if self.model_type == 'vicuna':
                chat_state = default_conversation.copy()
            else:
                chat_state = conv_llava_llama_2.copy()
            
            # 设置系统prompt
            chat_state.system = self.ghost_probing_prompt
            
            # 上传视频
            img_list = []
            try:
                # 使用chat.upload_video上传视频
                llm_message = self.chat.upload_video(video_path, chat_state, img_list)
                logger.info(f"✅ Video uploaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to upload video: {e}")
                return {
                    "error": f"Failed to upload video: {e}",
                    "ghost_probing_detected": False,
                    "processing_status": "failed"
                }
            
            # 询问鬼探头分析
            question = "Please analyze this video for ghost probing situations as requested."
            self.chat.ask(question, chat_state)
            
            # 获取回答
            try:
                response = self.chat.answer(
                    conv=chat_state,
                    img_list=img_list,
                    num_beams=1,
                    temperature=0.7,
                    max_new_tokens=500,
                    max_length=2000
                )[0]
                
                logger.info(f"✅ Got response from Video-LLaMA2")
                
                # 解析回答
                result = self._parse_response(response, video_name)
                result["video_path"] = video_path
                result["raw_response"] = response
                result["processing_status"] = "success"
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Failed to get response: {e}")
                return {
                    "error": f"Failed to get response: {e}",
                    "ghost_probing_detected": False,
                    "processing_status": "failed"
                }
        
        except Exception as e:
            logger.error(f"❌ Error analyzing video {video_path}: {e}")
            return {
                "error": str(e),
                "ghost_probing_detected": False,
                "processing_status": "failed"
            }
    
    def _parse_response(self, response: str, video_id: str) -> Dict:
        """
        解析Video-LLaMA2的回答
        
        Args:
            response: 模型的原始回答
            video_id: 视频ID
            
        Returns:
            解析后的结果字典
        """
        try:
            # 尝试从响应中提取JSON数组
            import re
            
            # 查找JSON数组格式的响应
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_segments = json.loads(json_str)
                    if isinstance(parsed_segments, list) and len(parsed_segments) > 0:
                        # 更新video_id
                        for segment in parsed_segments:
                            segment["video_id"] = video_id
                        
                        # 检测鬼探头
                        ghost_detected = False
                        ghost_timestamp = None
                        
                        for segment in parsed_segments:
                            key_actions = segment.get("key_actions", "").lower()
                            if "ghost probing" in key_actions:
                                ghost_detected = True
                                ghost_timestamp = segment.get("Start_Timestamp", "unknown")
                                break
                        
                        return {
                            "segments": parsed_segments,
                            "ghost_probing_detected": ghost_detected,
                            "time_of_occurrence": ghost_timestamp if ghost_detected else "none",
                            "total_segments": len(parsed_segments),
                            "parsing_success": True
                        }
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parsing error: {e}")
            
            # 如果没有找到JSON数组，尝试查找单个JSON对象
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed = json.loads(json_str)
                    # 转换为segment格式
                    segment = {
                        "video_id": video_id,
                        "segment_id": "segment_000",
                        "Start_Timestamp": "0.0s",
                        "End_Timestamp": "unknown",
                        "sentiment": parsed.get("sentiment", "Neutral"),
                        "scene_theme": parsed.get("scene_theme", "Routine"),
                        "characters": parsed.get("characters", "Not specified"),
                        "summary": parsed.get("description", response[:200]),
                        "actions": parsed.get("actions", "Not specified"),
                        "key_objects": parsed.get("key_objects", "Not specified"),
                        "key_actions": parsed.get("key_actions", "Not specified"),
                        "next_action": parsed.get("next_action", {
                            "speed_control": "maintain speed",
                            "direction_control": "keep direction",
                            "lane_control": "maintain current lane"
                        })
                    }
                    
                    # 检测鬼探头
                    ghost_detected = "ghost probing" in segment["key_actions"].lower()
                    
                    return {
                        "segments": [segment],
                        "ghost_probing_detected": ghost_detected,
                        "time_of_occurrence": segment["Start_Timestamp"] if ghost_detected else "none",
                        "total_segments": 1,
                        "parsing_success": True
                    }
                    
                except json.JSONDecodeError:
                    pass
            
            # 如果无法解析JSON，创建基于文本的segment
            response_lower = response.lower()
            
            # 检测鬼探头相关关键词
            ghost_keywords = [
                "ghost probing", "sudden appearance", "emergency braking",
                "close distance", "blind spot", "dangerous", "immediate response"
            ]
            
            ghost_detected = any(keyword in response_lower for keyword in ghost_keywords)
            
            segment = {
                "video_id": video_id,
                "segment_id": "segment_000",
                "Start_Timestamp": "0.0s",
                "End_Timestamp": "unknown",
                "sentiment": "Negative" if ghost_detected else "Neutral",
                "scene_theme": "Dangerous" if ghost_detected else "Routine",
                "characters": "Not specified",
                "summary": response[:300] if len(response) > 300 else response,
                "actions": "Emergency response required" if ghost_detected else "Normal driving",
                "key_objects": "Not specified",
                "key_actions": "ghost probing" if ghost_detected else "normal traffic flow",
                "next_action": {
                    "speed_control": "rapid deceleration" if ghost_detected else "maintain speed",
                    "direction_control": "keep direction",
                    "lane_control": "maintain current lane"
                }
            }
            
            return {
                "segments": [segment],
                "ghost_probing_detected": ghost_detected,
                "time_of_occurrence": "0.0s" if ghost_detected else "none",
                "total_segments": 1,
                "parsing_success": False,
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"❌ Error parsing response: {e}")
            
            # 创建错误处理的segment
            segment = {
                "video_id": video_id,
                "segment_id": "segment_000",
                "Start_Timestamp": "0.0s",
                "End_Timestamp": "unknown",
                "sentiment": "Neutral",
                "scene_theme": "Routine",
                "characters": "Processing error",
                "summary": f"Error parsing response: {str(e)}",
                "actions": "Processing failed",
                "key_objects": "Not available",
                "key_actions": "processing error",
                "next_action": {
                    "speed_control": "maintain speed",
                    "direction_control": "keep direction",
                    "lane_control": "maintain current lane"
                }
            }
            
            return {
                "segments": [segment],
                "ghost_probing_detected": False,
                "time_of_occurrence": "none",
                "total_segments": 1,
                "parsing_success": False,
                "error": str(e)
            }
    
    def batch_analyze(self, video_folder: str, max_videos: int = 100) -> List[Dict]:
        """
        批量分析视频
        
        Args:
            video_folder: 视频文件夹路径
            max_videos: 最大处理视频数量
            
        Returns:
            分析结果列表
        """
        try:
            # 获取目标视频列表
            video_folder_path = Path(video_folder)
            target_videos = []
            
            # 获取images_1_001到images_5_XXX的视频
            for i in range(1, 6):
                pattern = f"images_{i}_*.avi"
                videos = sorted(video_folder_path.glob(pattern))
                target_videos.extend(videos)
                if len(target_videos) >= max_videos:
                    break
            
            target_videos = target_videos[:max_videos]
            
            logger.info(f"📊 Starting batch analysis of {len(target_videos)} videos")
            
            results = []
            
            with tqdm(total=len(target_videos), desc="Processing videos") as pbar:
                for i, video_path in enumerate(target_videos):
                    try:
                        video_name = video_path.name
                        logger.info(f"🎬 Processing {i+1}/{len(target_videos)}: {video_name}")
                        
                        # 分析视频
                        result = self.analyze_video(str(video_path))
                        result["video_id"] = video_name
                        result["video_index"] = i + 1
                        
                        results.append(result)
                        
                        # 更新进度条
                        pbar.set_postfix({
                            "current": video_name,
                            "ghost_detected": result.get("ghost_probing_detected", False)
                        })
                        pbar.update(1)
                        
                        # 每10个视频保存一次中间结果
                        if (i + 1) % 10 == 0:
                            self._save_intermediate_results(results, i + 1)
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing video {video_path}: {e}")
                        results.append({
                            "video_id": video_path.name,
                            "video_index": i + 1,
                            "error": str(e),
                            "ghost_probing_detected": False,
                            "processing_status": "failed"
                        })
                        pbar.update(1)
            
            logger.info(f"✅ Batch analysis completed: {len(results)} videos processed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch analysis failed: {e}")
            return []
    
    def _save_intermediate_results(self, results: List[Dict], count: int):
        """保存中间结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"intermediate_results_{count}_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Intermediate results saved: {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save intermediate results: {e}")
    
    def format_for_comparison(self, results: List[Dict], groundtruth_file: str) -> pd.DataFrame:
        """
        格式化结果以便与ground truth比较
        
        Args:
            results: 分析结果列表
            groundtruth_file: Ground truth文件路径
            
        Returns:
            格式化的DataFrame
        """
        try:
            # 加载ground truth
            gt_df = pd.read_csv(groundtruth_file, sep='\t')
            gt_dict = dict(zip(gt_df['video_id'], gt_df['ground_truth_label']))
            
            # 格式化结果
            formatted_results = []
            for result in results:
                video_id = result.get("video_id", "unknown")
                
                # 格式化预测标签
                if result.get("ghost_probing_detected", False):
                    time_str = result.get("time_of_occurrence", "unknown")
                    if time_str != "none" and time_str != "unknown":
                        predicted_label = f"{time_str}: ghost probing"
                    else:
                        predicted_label = "ghost probing"
                else:
                    predicted_label = "none"
                
                # 获取ground truth
                ground_truth = gt_dict.get(video_id, "unknown")
                
                formatted_results.append({
                    "video_id": video_id,
                    "predicted_label": predicted_label,
                    "ground_truth_label": ground_truth,
                    "processing_status": result.get("processing_status", "unknown"),
                    "danger_level": result.get("danger_level", 0),
                    "object_type": result.get("object_type", "none"),
                    "description": result.get("description", ""),
                    "raw_response": result.get("raw_response", "")
                })
            
            return pd.DataFrame(formatted_results)
            
        except Exception as e:
            logger.error(f"❌ Error formatting results: {e}")
            return pd.DataFrame()
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """
        计算性能指标
        
        Args:
            df: 包含预测和ground truth的DataFrame
            
        Returns:
            性能指标字典
        """
        try:
            # 统计各种情况
            tp = 0  # True Positive
            fp = 0  # False Positive
            tn = 0  # True Negative
            fn = 0  # False Negative
            
            successful_results = df[df['processing_status'] == 'success']
            
            for _, row in successful_results.iterrows():
                predicted = row['predicted_label']
                ground_truth = row['ground_truth_label']
                
                # 判断是否为鬼探头
                predicted_ghost = 'ghost probing' in predicted.lower() if predicted != 'none' else False
                ground_truth_ghost = 'ghost probing' in ground_truth.lower() if ground_truth != 'none' else False
                
                if predicted_ghost and ground_truth_ghost:
                    tp += 1
                elif predicted_ghost and not ground_truth_ghost:
                    fp += 1
                elif not predicted_ghost and ground_truth_ghost:
                    fn += 1
                else:
                    tn += 1
            
            # 计算指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'total_videos': len(df),
                'successful_processing': len(successful_results),
                'failed_processing': len(df) - len(successful_results),
                'true_positive': tp,
                'false_positive': fp,
                'true_negative': tn,
                'false_negative': fn,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            return {}
    
    def save_results(self, results: List[Dict], df: pd.DataFrame, metrics: Dict):
        """
        保存分析结果
        
        Args:
            results: 原始结果列表
            df: 格式化的DataFrame
            metrics: 性能指标
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存详细JSON结果
            json_file = f"video_llama2_ghost_probing_results_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存分段级别的详细结果 (类似ActionSummary格式)
            segments_file = f"video_llama2_segments_{timestamp}.json"
            all_segments = []
            
            for result in results:
                if "segments" in result and result["segments"]:
                    all_segments.extend(result["segments"])
            
            with open(segments_file, 'w', encoding='utf-8') as f:
                json.dump(all_segments, f, ensure_ascii=False, indent=2)
            
            # 为每个视频保存单独的ActionSummary格式文件
            segments_dir = f"video_llama2_segments_{timestamp}"
            os.makedirs(segments_dir, exist_ok=True)
            
            for result in results:
                if "segments" in result and result["segments"]:
                    video_id = result.get("video_id", "unknown")
                    video_segments_file = os.path.join(segments_dir, f"actionSummary_{video_id}.json")
                    
                    with open(video_segments_file, 'w', encoding='utf-8') as f:
                        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
            
            # 保存对比CSV
            csv_file = f"video_llama2_ghost_probing_comparison_{timestamp}.csv"
            df.to_csv(csv_file, sep='\t', index=False, encoding='utf-8')
            
            # 保存性能指标
            metrics_file = f"video_llama2_ghost_probing_metrics_{timestamp}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Results saved:")
            logger.info(f"  - 原始结果JSON: {json_file}")
            logger.info(f"  - 所有分段JSON: {segments_file}")
            logger.info(f"  - 单视频分段目录: {segments_dir}")
            logger.info(f"  - 对比CSV: {csv_file}")
            logger.info(f"  - 性能指标: {metrics_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Video-LLaMA2 Ghost Probing Detection')
    parser.add_argument('--config', default='eval_configs/video_llama_eval_withaudio.yaml', 
                      help='Video-LLaMA2 config file path')
    parser.add_argument('--model-type', default='llama_v2', choices=['llama_v2', 'vicuna'],
                      help='Model type')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--video-folder', default='../../DADA-2000-videos', 
                      help='Video folder path')
    parser.add_argument('--groundtruth-file', default='../../result/groundtruth_labels.csv',
                      help='Ground truth file path')
    parser.add_argument('--max-videos', type=int, default=100, help='Maximum videos to process')
    parser.add_argument('--single-video', help='Process single video file')
    parser.add_argument('--dry-run', action='store_true', help='Preview videos without processing')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = VideoLLaMA2GhostProbingDetector(
        config_path=args.config,
        model_type=args.model_type,
        gpu_id=args.gpu_id
    )
    
    # 初始化模型
    if not detector.initialize_model():
        logger.error("❌ Failed to initialize model")
        return
    
    if args.dry_run:
        logger.info("🔍 Dry run mode - previewing videos")
        video_folder_path = Path(args.video_folder)
        target_videos = []
        for i in range(1, 6):
            pattern = f"images_{i}_*.avi"
            videos = sorted(video_folder_path.glob(pattern))
            target_videos.extend(videos)
            if len(target_videos) >= args.max_videos:
                break
        target_videos = target_videos[:args.max_videos]
        
        logger.info(f"Will process {len(target_videos)} videos:")
        for i, video in enumerate(target_videos):
            logger.info(f"  {i+1:3d}. {video.name}")
        return
    
    if args.single_video:
        logger.info(f"🎬 Processing single video: {args.single_video}")
        result = detector.analyze_video(args.single_video)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    # 批量处理
    logger.info("🚀 Starting batch processing")
    results = detector.batch_analyze(args.video_folder, args.max_videos)
    
    # 格式化结果
    logger.info("📊 Formatting results for comparison")
    df = detector.format_for_comparison(results, args.groundtruth_file)
    
    # 计算指标
    logger.info("📈 Calculating performance metrics")
    metrics = detector.calculate_metrics(df)
    
    # 保存结果
    logger.info("💾 Saving results")
    detector.save_results(results, df, metrics)
    
    # 输出总结
    logger.info("=" * 60)
    logger.info("📋 FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total videos processed: {metrics.get('total_videos', 0)}")
    logger.info(f"Successful processing: {metrics.get('successful_processing', 0)}")
    logger.info(f"Failed processing: {metrics.get('failed_processing', 0)}")
    logger.info(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
    logger.info(f"Precision: {metrics.get('precision', 0):.3f}")
    logger.info(f"Recall: {metrics.get('recall', 0):.3f}")
    logger.info(f"F1 Score: {metrics.get('f1_score', 0):.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()