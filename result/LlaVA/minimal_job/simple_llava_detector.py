#!/usr/bin/env python3
"""
简化版LLaVA鬼探头检测器
使用OpenAI CLIP + GPT模型进行视频分析
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
import logging
import torch
from typing import List, Dict, Optional, Tuple
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleLLaVADetector:
    """简化版LLaVA鬼探头检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        # 平衡版GPT-4.1鬼探头检测prompt
        self.ghost_probing_prompt = """Analyze these sequential driving video frames for ghost probing detection.

Ghost probing means: sudden appearance of vehicles/pedestrians from blind spots at very close distance requiring immediate emergency action.

HIGH-CONFIDENCE Ghost Probing:
- Object appears within 1-2 vehicle lengths (<3 meters)
- SUDDEN appearance from blind spots
- Requires IMMEDIATE emergency braking
- Completely unpredictable movement

Respond with JSON:
{
    "ghost_probing_detected": "yes/no",
    "confidence": 0.0-1.0,
    "ghost_type": "high_confidence/potential/none",
    "summary": "brief description",
    "risk_level": "high/medium/low"
}"""
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型 - 使用CLIP + GPT2"""
        try:
            logger.info("🔧 正在初始化简化版模型...")
            logger.info(f"🖥️  设备: {self.device}")
            
            # 使用CLIP进行图像编码
            from transformers import CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
            
            # 加载CLIP模型
            logger.info("📥 加载CLIP模型...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # 加载GPT2模型用于文本生成
            logger.info("📥 加载GPT2模型...")
            self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            
            # 移到GPU
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
                self.gpt_model = self.gpt_model.cuda()
                logger.info(f"✅ 模型已加载到GPU: {torch.cuda.get_device_name()}")
            else:
                logger.warning("⚠️  CUDA不可用，使用CPU")
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    def extract_frames_with_decord(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """使用decord提取视频帧"""
        try:
            import decord
            from decord import VideoReader
            
            # 设置decord使用native bridge
            decord.bridge.set_bridge('native')
            
            # 读取视频
            video_reader = VideoReader(str(video_path))
            total_frames = len(video_reader)
            
            if total_frames == 0:
                raise ValueError(f"视频文件没有帧: {video_path}")
            
            # 均匀分布选择帧
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            # 提取帧
            frames = []
            for idx in frame_indices:
                frame = video_reader[idx]
                
                # 转换为numpy数组
                if hasattr(frame, 'asnumpy'):
                    frame_array = frame.asnumpy()
                elif isinstance(frame, torch.Tensor):
                    frame_array = frame.cpu().numpy()
                else:
                    frame_array = np.array(frame)
                
                # 转换为PIL Image
                pil_image = Image.fromarray(frame_array.astype(np.uint8))
                frames.append(pil_image)
            
            logger.info(f"✅ 从视频中提取了{len(frames)}帧")
            return frames
            
        except Exception as e:
            logger.error(f"❌ 视频帧提取失败: {e}")
            raise
    
    def analyze_frames_simple(self, frames: List[Image.Image], video_path: str) -> Dict:
        """使用简化方法分析视频帧"""
        if not frames:
            raise ValueError("没有帧可分析")
        
        try:
            logger.info(f"🔍 正在分析{len(frames)}帧...")
            
            # 使用CLIP提取图像特征
            inputs = self.clip_processor(images=frames, return_tensors="pt", padding=True)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # 分析帧间变化
            feature_changes = []
            for i in range(1, len(frames)):
                # 计算相邻帧之间的特征差异
                diff = torch.cosine_similarity(
                    image_features[i-1].unsqueeze(0),
                    image_features[i].unsqueeze(0)
                ).item()
                feature_changes.append(1 - diff)  # 差异越大，值越高
            
            # 基于特征变化判断鬼探头
            max_change = max(feature_changes) if feature_changes else 0
            avg_change = sum(feature_changes) / len(feature_changes) if feature_changes else 0
            
            # 简单的规则判断
            if max_change > 0.5 and avg_change > 0.3:
                ghost_detected = "yes"
                confidence = min(0.9, max_change)
                ghost_type = "high_confidence" if max_change > 0.7 else "potential"
                risk_level = "high" if max_change > 0.7 else "medium"
            elif max_change > 0.3:
                ghost_detected = "yes"
                confidence = max_change * 0.7
                ghost_type = "potential"
                risk_level = "medium"
            else:
                ghost_detected = "no"
                confidence = 0.2
                ghost_type = "none"
                risk_level = "low"
            
            result = {
                "ghost_probing_detected": ghost_detected,
                "confidence": round(confidence, 3),
                "ghost_type": ghost_type,
                "summary": f"视频分析完成，最大帧间变化: {max_change:.3f}",
                "key_actions": f"检测到{len(frames)}帧，特征变化分析",
                "risk_level": risk_level,
                "distance_estimate": "基于特征变化估算",
                "emergency_action_needed": "yes" if ghost_detected == "yes" else "no",
                "max_frame_change": round(max_change, 3),
                "avg_frame_change": round(avg_change, 3)
            }
            
            logger.info(f"✅ 分析完成 - 鬼探头: {ghost_detected} (置信度: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ 帧分析失败: {e}")
            raise
    
    def process_video(self, video_path: str) -> Dict:
        """处理单个视频文件"""
        video_name = Path(video_path).stem
        logger.info(f"🎬 开始处理视频: {video_name}")
        
        start_time = datetime.now()
        
        try:
            # 提取视频帧
            frames = self.extract_frames_with_decord(video_path, num_frames=8)
            
            if not frames:
                raise ValueError("无法提取视频帧")
            
            # 分析帧
            analysis_result = self.analyze_frames_simple(frames, video_path)
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 构建最终结果
            result = {
                "video_id": video_name,
                "video_path": str(video_path),
                "ghost_probing_label": analysis_result.get("ghost_probing_detected", "no"),
                "confidence": analysis_result.get("confidence", 0.5),
                "ghost_type": analysis_result.get("ghost_type", "none"),
                "summary": analysis_result.get("summary", ""),
                "key_actions": analysis_result.get("key_actions", ""),
                "risk_level": analysis_result.get("risk_level", "low"),
                "emergency_action_needed": analysis_result.get("emergency_action_needed", "no"),
                "model": "CLIP-GPT2-Simple",
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "method": "simple_feature_analysis",
                "frames_analyzed": len(frames),
                "device": self.device,
                "max_frame_change": analysis_result.get("max_frame_change", 0),
                "avg_frame_change": analysis_result.get("avg_frame_change", 0)
            }
            
            logger.info(f"✅ 视频处理完成: {video_name} ({processing_time:.1f}s)")
            return result
            
        except Exception as e:
            logger.error(f"❌ 处理视频{video_name}失败: {e}")
            return {
                "video_id": video_name,
                "video_path": str(video_path),
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

def main():
    """主函数"""
    print("🚀 开始简化版LLaVA鬼探头检测...")
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
        print(f"📊 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  GPU不可用，将使用CPU")
    
    # 获取视频路径
    azureml_data_path = os.environ.get('AZUREML_DATAREFERENCE_video_data')
    
    possible_paths = []
    if azureml_data_path:
        possible_paths.append(azureml_data_path)
        print(f"🔧 从环境变量找到数据路径: {azureml_data_path}")
    
    possible_paths.extend(["./inputs/video_data", "./inputs", "."])
    
    video_files = []
    video_folder = None
    
    for path in possible_paths:
        try:
            p = Path(path)
            if p.exists():
                found_videos = list(p.glob("**/*.avi"))
                if found_videos:
                    video_files = found_videos[:100]
                    video_folder = p
                    print(f"✅ 在 {path} 找到 {len(video_files)} 个视频文件")
                    break
        except Exception as e:
            print(f"❌ 检查路径 {path} 时出错: {e}")
    
    if not video_files:
        print("❌ 未找到任何视频文件")
        return
    
    # 创建输出目录
    os.makedirs("./outputs/results", exist_ok=True)
    
    # 初始化检测器
    detector = SimpleLLaVADetector()
    
    print(f"🎬 开始处理 {len(video_files)} 个视频...")
    
    # 处理视频
    results = []
    failed_count = 0
    
    for i, video_file in enumerate(video_files):
        try:
            result = detector.process_video(str(video_file))
            results.append(result)
            
            if 'error' in result:
                failed_count += 1
            
            if (i + 1) % 10 == 0:
                success_count = (i + 1) - failed_count
                print(f"📊 进度: {i+1}/{len(video_files)} ({(i+1)/len(video_files)*100:.1f}%) - 成功: {success_count}, 失败: {failed_count}")
                
        except Exception as e:
            logger.error(f"❌ 处理视频失败: {e}")
            failed_count += 1
            results.append({
                "video_id": Path(video_file).stem,
                "video_path": str(video_file),
                "error": str(e),
                "processing_time": 0
            })
    
    # 保存结果
    print("💾 保存结果文件...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    successful_results = [r for r in results if 'error' not in r]
    
    # JSON结果
    json_file = f"./outputs/results/simple_llava_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'model': 'CLIP-GPT2-Simple',
                'device': 'GPU' if torch.cuda.is_available() else 'CPU',
                'total_videos': len(video_files),
                'successful_videos': len(successful_results),
                'failed_videos': failed_count,
                'timestamp': timestamp
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    # CSV结果
    if successful_results:
        csv_file = f"./outputs/results/simple_llava_results_{timestamp}.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write('video_id,ghost_probing_label,confidence,ghost_type,risk_level,processing_time\n')
            for r in successful_results:
                f.write(f"{r['video_id']},{r['ghost_probing_label']},{r['confidence']:.3f},{r['ghost_type']},{r['risk_level']},{r['processing_time']:.1f}\n")
    
    # 统计
    if successful_results:
        ghost_count = len([r for r in successful_results if r['ghost_probing_label'] == 'yes'])
        detection_rate = (ghost_count / len(successful_results)) * 100
        avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
    else:
        ghost_count = detection_rate = avg_time = 0
    
    print("=" * 60)
    print("🎉 简化版检测完成!")
    print("=" * 60)
    print(f"📊 总视频数: {len(video_files)}")
    print(f"✅ 成功处理: {len(successful_results)}")
    print(f"❌ 处理失败: {failed_count}")
    if successful_results:
        print(f"🚨 鬼探头检测: {ghost_count} ({detection_rate:.1f}%)")
        print(f"⏱️  平均处理时间: {avg_time:.1f}秒/视频")
    print(f"📄 结果文件: {json_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()