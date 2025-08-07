#!/usr/bin/env python3
"""
DriveMM DADA-2000 Video Analysis Script
专用于DADA-2000数据集的DriveMM推理脚本

基于DriveMM: All-in-One Large Multimodal Model for Autonomous Driving
论文: https://arxiv.org/abs/2412.07689
代码: https://github.com/zhijian11/DriveMM
"""

import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import logging
import argparse
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.train.train import preprocess_llama3
    DRIVEMM_AVAILABLE = True
except ImportError:
    DRIVEMM_AVAILABLE = False
    logger.warning("DriveMM dependencies not available. Running in demo mode.")

class DriveMM_DADA2000_Analyzer:
    """DriveMM DADA-2000 视频分析器"""
    
    def __init__(self, model_path: str = "./ckpt/DriveMM", device: str = "auto"):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        
        # DADA-2000特定的分析提示词
        self.dada_prompts = {
            "ghost_probing": "<image>\nAnalyze this driving scene for potential ghost probing incidents. A ghost probing occurs when a pedestrian or cyclist suddenly appears from behind an obstacle (like a parked car, building corner, or blind spot) into the vehicle's path. Look carefully for: 1) Pedestrians or cyclists near parked vehicles, 2) Movement from behind obstacles, 3) Sudden appearances in the vehicle's path. Respond with: 'GHOST_PROBING_DETECTED' if you see evidence of ghost probing, or 'NO_GHOST_PROBING' if the scene appears normal. Then explain your reasoning.",
            
            "scene_analysis": "<image>\nAnalyze this autonomous driving scene and provide: 1) Scene type (urban/highway/intersection/residential), 2) Key objects (vehicles, pedestrians, traffic signs, etc.), 3) Risk level (low/medium/high), 4) Potential hazards, 5) Driving recommendations.",
            
            "object_detection": "<image>\nIdentify and describe all important objects in this driving scene: vehicles (cars, trucks, buses), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic infrastructure (signs, lights, lanes), and any obstacles or hazards. For each object, describe its position, direction, and relevance to driving decisions.",
            
            "risk_assessment": "<image>\nEvaluate the risk level of this driving scenario. Consider: traffic density, pedestrian activity, road conditions, visibility, potential conflicts. Provide a risk score (1-10) and explain the main safety concerns.",
            
            "driving_advice": "<image>\nAs an autonomous driving system, what actions should the ego vehicle take in this situation? Consider speed adjustments, steering, lane changes, stopping, and explain the reasoning for each recommendation."
        }
        
        if DRIVEMM_AVAILABLE:
            self._load_model()
    
    def _setup_device(self, device: str):
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_model(self):
        """加载DriveMM模型"""
        if not os.path.exists(self.model_path):
            logger.error(f"Model path {self.model_path} does not exist!")
            return False
            
        try:
            logger.info(f"Loading DriveMM model from {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
            model_name = 'llama'
            llava_model_args = {"multimodal": True}
            
            self.tokenizer, self.model, self.image_processor, max_length = load_pretrained_model(
                self.model_path, None, model_name, device_map=self.device, **llava_model_args
            )
            
            self.model.eval()
            logger.info("DriveMM model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load DriveMM model: {e}")
            return False
    
    def extract_video_frames(self, video_path: str, num_frames: int = 5) -> List[Image.Image]:
        """从视频中提取关键帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb).convert("RGB")
                frames.append(pil_image)
        
        cap.release()
        
        if len(frames) < num_frames:
            # 如果帧数不足，复制最后一帧
            while len(frames) < num_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    # 创建空白帧
                    blank_frame = Image.new('RGB', (640, 480), color=(0, 0, 0))
                    frames.append(blank_frame)
        
        return frames[:num_frames]
    
    def analyze_with_drivemm(self, images: List[Image.Image], prompt: str) -> str:
        """使用DriveMM分析图像"""
        if not DRIVEMM_AVAILABLE or self.model is None:
            return self._demo_response(prompt)
        
        try:
            # 处理图像
            image_tensors = process_images(images, self.image_processor, self.model.config)
            image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
            
            # 准备输入
            sources = [[{"from": 'human', "value": prompt}, {"from": 'gpt', "value": ''}]]
            input_ids = preprocess_llama3(sources, self.tokenizer, has_image=True)['input_ids'][:, :-1].to(self.device)
            
            image_sizes = [image.size for image in images]
            
            # 推理
            with torch.no_grad():
                cont = self.model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=1024,
                    modalities=['video' if len(images) > 1 else 'image']
                )
            
            # 解码输出
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            return text_outputs[0] if text_outputs else "No response generated"
            
        except Exception as e:
            logger.error(f"Error during DriveMM inference: {e}")
            return f"Error: {str(e)}"
    
    def _demo_response(self, prompt: str) -> str:
        """演示模式的响应"""
        if "ghost probing" in prompt.lower():
            return "NO_GHOST_PROBING - Demo mode: Cannot perform real analysis without DriveMM model. This is a placeholder response."
        elif "scene analysis" in prompt.lower():
            return "Demo mode: Scene appears to be an urban driving environment with moderate traffic. Risk level: Medium. Real analysis requires DriveMM model."
        elif "risk assessment" in prompt.lower():
            return "Demo mode: Risk score: 5/10. Real risk assessment requires DriveMM model and proper image analysis."
        else:
            return "Demo mode: This is a placeholder response. Real analysis requires DriveMM model to be properly loaded."
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """分析单个视频"""
        video_id = Path(video_path).stem
        logger.info(f"Analyzing video: {video_id}")
        
        start_time = datetime.now()
        
        try:
            # 提取视频帧
            frames = self.extract_video_frames(video_path, num_frames=5)
            
            # 进行多项分析
            results = {
                "video_id": video_id,
                "video_path": video_path,
                "timestamp": start_time.isoformat(),
                "analysis_results": {}
            }
            
            # 1. 鬼探头检测（DADA-2000核心任务）
            ghost_analysis = self.analyze_with_drivemm(frames, self.dada_prompts["ghost_probing"])
            ghost_detected = "GHOST_PROBING_DETECTED" in ghost_analysis.upper()
            
            results["analysis_results"]["ghost_probing"] = {
                "detected": ghost_detected,
                "analysis": ghost_analysis,
                "confidence": "high" if ghost_detected else "n/a"
            }
            
            # 2. 场景分析
            scene_analysis = self.analyze_with_drivemm(frames, self.dada_prompts["scene_analysis"])
            results["analysis_results"]["scene_analysis"] = {
                "description": scene_analysis
            }
            
            # 3. 风险评估
            risk_analysis = self.analyze_with_drivemm(frames, self.dada_prompts["risk_assessment"])
            results["analysis_results"]["risk_assessment"] = {
                "assessment": risk_analysis
            }
            
            # 4. 驾驶建议
            driving_advice = self.analyze_with_drivemm(frames, self.dada_prompts["driving_advice"])
            results["analysis_results"]["driving_advice"] = {
                "recommendations": driving_advice
            }
            
            end_time = datetime.now()
            results["processing_time_seconds"] = (end_time - start_time).total_seconds()
            
            # 输出结果摘要
            status = "🚨 GHOST PROBING DETECTED" if ghost_detected else "✅ NO GHOST PROBING"
            logger.info(f"  {status}")
            logger.info(f"  Processing time: {results['processing_time_seconds']:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_id}: {str(e)}")
            return {
                "video_id": video_id,
                "video_path": video_path,
                "error": str(e),
                "timestamp": start_time.isoformat()
            }
    
    def batch_analyze_dada2000(self, video_dir: str, output_dir: str = "drivemm_results", limit: int = None):
        """批量分析DADA-2000数据集"""
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 获取视频文件
        video_files = list(video_dir.glob("*.avi"))
        if limit:
            video_files = video_files[:limit]
        
        logger.info(f"Found {len(video_files)} videos to analyze")
        logger.info(f"Model available: {DRIVEMM_AVAILABLE and self.model is not None}")
        
        results = []
        ghost_detections = 0
        
        for idx, video_file in enumerate(video_files, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {idx}/{len(video_files)}: {video_file.name}")
            logger.info(f"{'='*50}")
            
            # 分析视频
            result = self.analyze_video(str(video_file))
            
            if "error" not in result:
                results.append(result)
                
                # 统计鬼探头检测
                if result["analysis_results"]["ghost_probing"]["detected"]:
                    ghost_detections += 1
                
                # 保存单个结果
                result_file = output_dir / f"drivemm_{result['video_id']}.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
        
        # 生成批量分析报告
        summary = {
            "drivemm_analysis_summary": {
                "total_videos": len(results),
                "ghost_probing_detected": ghost_detections,
                "detection_rate": ghost_detections / len(results) if results else 0,
                "model_available": DRIVEMM_AVAILABLE and self.model is not None,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": results
        }
        
        # 保存总结报告
        summary_file = output_dir / "drivemm_batch_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n🎉 DRIVEMM ANALYSIS COMPLETED!")
        logger.info(f"="*50)
        logger.info(f"Videos analyzed: {len(results)}")
        logger.info(f"Ghost probing detected: {ghost_detections}")
        logger.info(f"Detection rate: {ghost_detections / len(results):.1%}" if results else "N/A")
        logger.info(f"Results saved to: {output_dir}")
        
        return summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DriveMM DADA-2000 Video Analysis")
    parser.add_argument("--video_dir", default="/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos", 
                       help="DADA-2000 video directory")
    parser.add_argument("--model_path", default="./ckpt/DriveMM", 
                       help="DriveMM model path")
    parser.add_argument("--output_dir", default="drivemm_results", 
                       help="Output directory")
    parser.add_argument("--limit", type=int, default=5, 
                       help="Limit number of videos to process")
    parser.add_argument("--device", default="auto", 
                       help="Device to use (auto/cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    print("🚀 DRIVEMM DADA-2000 VIDEO ANALYSIS")
    print("="*50)
    print(f"📄 Paper: https://arxiv.org/abs/2412.07689")
    print(f"💻 Code: https://github.com/zhijian11/DriveMM")
    print(f"🎯 Task: DADA-2000 Ghost Probing Analysis")
    
    # 初始化分析器
    analyzer = DriveMM_DADA2000_Analyzer(
        model_path=args.model_path,
        device=args.device
    )
    
    # 检查视频目录
    if not os.path.exists(args.video_dir):
        print(f"❌ Video directory not found: {args.video_dir}")
        print("🎭 Running in demo mode...")
        args.video_dir = "./scripts/inference_demo/bddx"  # 使用demo视频
    
    # 执行批量分析
    summary = analyzer.batch_analyze_dada2000(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        limit=args.limit
    )
    
    print(f"\n📊 Analysis Summary:")
    print(f"   Model Available: {summary['drivemm_analysis_summary']['model_available']}")
    print(f"   Videos Processed: {summary['drivemm_analysis_summary']['total_videos']}")
    print(f"   Ghost Detections: {summary['drivemm_analysis_summary']['ghost_probing_detected']}")
    print(f"📁 Results: {args.output_dir}/drivemm_batch_summary.json")

if __name__ == "__main__":
    main()