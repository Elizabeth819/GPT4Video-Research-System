#!/usr/bin/env python3
"""
GPU显存监控版LLaVA检测器
监控整个处理过程中的GPU显存消耗
"""

import json
import os
import gc
from pathlib import Path
from datetime import datetime
from PIL import Image
import logging
import torch
from typing import List, Dict
import numpy as np
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gpu_memory_info():
    """获取GPU显存信息 (MB)"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024 / 1024,  # MB
            'reserved': torch.cuda.memory_reserved() / 1024 / 1024,   # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

def log_memory_usage(stage: str):
    """记录当前阶段的显存使用情况"""
    mem_info = get_gpu_memory_info()
    logger.info(f"🖥️  [{stage}] GPU显存 - 已分配: {mem_info['allocated']:.1f}MB, "
                f"已保留: {mem_info['reserved']:.1f}MB, 峰值: {mem_info['max_allocated']:.1f}MB")
    return mem_info

class GPUMemoryLLaVADetector:
    """带GPU显存监控的LLaVA检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.memory_logs = []  # 记录显存使用历史
        
        log_memory_usage("初始化开始")
        self._initialize_model()
        log_memory_usage("初始化完成")
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            logger.info("🔧 正在初始化GPU显存监控版模型...")
            logger.info(f"🖥️  设备: {self.device}")
            
            if torch.cuda.is_available():
                # 重置GPU显存统计
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                log_memory_usage("缓存清理后")
            
            # 使用CLIP进行图像编码
            from transformers import CLIPProcessor, CLIPModel
            
            logger.info("📥 正在加载CLIP模型...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            log_memory_usage("CLIP模型加载后")
            
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            log_memory_usage("CLIP处理器加载后")
            
            # 移到GPU
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
                log_memory_usage("模型移到GPU后")
                logger.info(f"✅ 模型已加载到GPU: {torch.cuda.get_device_name()}")
            else:
                logger.warning("⚠️  CUDA不可用，使用CPU")
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    def extract_frames_with_memory_monitor(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """带显存监控的帧提取"""
        
        logger.info(f"🎬 开始处理视频: {Path(video_path).name}")
        mem_start = log_memory_usage("视频处理开始")
        
        try:
            import decord
            from decord import VideoReader
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                log_memory_usage("清理后")
            
            # 设置decord
            decord.bridge.set_bridge('native')
            
            # 读取视频
            logger.info("📖 正在读取视频...")
            video_reader = VideoReader(str(video_path))
            total_frames = len(video_reader)
            log_memory_usage("视频读取后")
            
            logger.info(f"📊 总帧数: {total_frames}")
            
            # 均匀分布选择帧
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            logger.info(f"📊 选择帧索引: {frame_indices.tolist()}")
            
            # 提取帧
            frames = []
            for i, idx in enumerate(frame_indices):
                logger.info(f"🔍 提取第 {i+1}/{num_frames} 帧 (索引: {idx})")
                
                # 获取帧前显存状态
                mem_before = log_memory_usage(f"帧{i+1}提取前")
                
                # 获取帧
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
                
                # 获取帧后显存状态
                mem_after = log_memory_usage(f"帧{i+1}提取后")
                
                # 记录显存变化
                mem_change = mem_after['allocated'] - mem_before['allocated']
                if abs(mem_change) > 0.1:  # 只记录显著变化
                    logger.info(f"  📊 帧{i+1}显存变化: {mem_change:+.1f}MB")
                
                # 清理临时对象
                del frame, frame_array
                gc.collect()
            
            mem_end = log_memory_usage("帧提取完成")
            total_mem_change = mem_end['allocated'] - mem_start['allocated']
            logger.info(f"📊 视频处理总显存变化: {total_mem_change:+.1f}MB")
            
            return frames
            
        except Exception as e:
            logger.error(f"❌ 帧提取失败: {e}")
            raise
    
    def analyze_frames_with_memory_monitor(self, frames: List[Image.Image]) -> Dict:
        """带显存监控的帧分析"""
        if not frames:
            raise ValueError("没有帧可分析")
        
        try:
            logger.info(f"🔍 开始分析{len(frames)}帧...")
            mem_start = log_memory_usage("CLIP分析开始")
            
            # CLIP处理
            logger.info("📝 CLIP预处理...")
            inputs = self.clip_processor(images=frames, return_tensors="pt", padding=True)
            mem_after_preprocess = log_memory_usage("CLIP预处理后")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                mem_after_cuda = log_memory_usage("数据移到GPU后")
            
            # CLIP推理
            logger.info("🧠 CLIP特征提取...")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            mem_after_inference = log_memory_usage("CLIP推理后")
            
            # 分析帧间变化
            feature_changes = []
            for i in range(1, len(frames)):
                diff = torch.cosine_similarity(
                    image_features[i-1].unsqueeze(0),
                    image_features[i].unsqueeze(0)
                ).item()
                feature_changes.append(1 - diff)
            
            max_change = max(feature_changes) if feature_changes else 0
            avg_change = sum(feature_changes) / len(feature_changes) if feature_changes else 0
            
            mem_final = log_memory_usage("分析完成")
            
            # 清理GPU显存
            del inputs, image_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            mem_after_cleanup = log_memory_usage("显存清理后")
            
            # 判断逻辑
            if max_change > 0.5:
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
            
            # 显存使用统计
            memory_stats = {
                'start_allocated': mem_start['allocated'],
                'peak_allocated': mem_after_inference['allocated'],
                'final_allocated': mem_final['allocated'],
                'cleanup_allocated': mem_after_cleanup['allocated'],
                'peak_reserved': mem_after_inference['reserved'],
                'memory_increase': mem_after_inference['allocated'] - mem_start['allocated'],
                'memory_freed': mem_final['allocated'] - mem_after_cleanup['allocated']
            }
            
            result = {
                "ghost_probing_detected": ghost_detected,
                "confidence": round(confidence, 3),
                "ghost_type": ghost_type,
                "summary": f"GPU显存监控分析完成，最大帧间变化: {max_change:.4f}",
                "key_actions": f"监控版检测{len(frames)}帧，CLIP特征分析",
                "risk_level": risk_level,
                "emergency_action_needed": "yes" if ghost_detected == "yes" else "no",
                "max_frame_change": round(max_change, 4),
                "avg_frame_change": round(avg_change, 4),
                "memory_stats": memory_stats
            }
            
            logger.info(f"📊 显存峰值: {memory_stats['peak_allocated']:.1f}MB")
            logger.info(f"📊 显存增长: {memory_stats['memory_increase']:+.1f}MB")
            logger.info(f"✅ 分析完成 - 鬼探头: {ghost_detected} (置信度: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 分析失败: {e}")
            raise
    
    def process_video(self, video_path: str) -> Dict:
        """处理单个视频"""
        video_name = Path(video_path).stem
        logger.info(f"🎬 开始处理视频: {video_name}")
        
        start_time = datetime.now()
        mem_initial = log_memory_usage("视频处理开始")
        
        try:
            # 提取视频帧
            frames = self.extract_frames_with_memory_monitor(video_path, num_frames=8)
            
            if not frames:
                raise ValueError("无法提取视频帧")
            
            # 分析帧
            analysis_result = self.analyze_frames_with_memory_monitor(frames)
            
            # 最终显存状态
            mem_final = log_memory_usage("视频处理完成")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 构建结果
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
                "model": "CLIP-GPU-Memory-Monitor",
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "frames_analyzed": len(frames),
                "device": self.device,
                "max_frame_change": analysis_result.get("max_frame_change", 0),
                "avg_frame_change": analysis_result.get("avg_frame_change", 0),
                "memory_stats": analysis_result.get("memory_stats", {}),
                "total_memory_change": mem_final['allocated'] - mem_initial['allocated']
            }
            
            logger.info(f"✅ 处理完成: {video_name} ({processing_time:.2f}s)")
            logger.info(f"📊 视频总显存变化: {result['total_memory_change']:+.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 处理失败: {video_name} - {e}")
            return {
                "video_id": video_name,
                "video_path": str(video_path),
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

def main():
    """GPU显存监控主函数"""
    print("🖥️  GPU显存监控版LLaVA鬼探头检测")
    print("=" * 60)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
        print(f"📊 GPU总显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f}MB")
        log_memory_usage("程序开始")
    else:
        print("⚠️  GPU不可用，将使用CPU")
        return
    
    # 获取视频路径
    azureml_data_path = os.environ.get('AZUREML_DATAREFERENCE_video_data')
    
    possible_paths = []
    if azureml_data_path:
        possible_paths.append(azureml_data_path)
        print(f"🔧 从环境变量找到数据路径: {azureml_data_path}")
    
    possible_paths.extend(["./inputs/video_data", "./inputs", "."])
    
    video_files = []
    for path in possible_paths:
        try:
            p = Path(path)
            if p.exists():
                found_videos = list(p.glob("**/*.avi"))
                if found_videos:
                    # 只取前3个视频进行显存监控测试
                    video_files = found_videos[:3]
                    print(f"✅ 在 {path} 找到 {len(found_videos)} 个视频，监控前 {len(video_files)} 个")
                    break
        except Exception as e:
            print(f"❌ 检查路径 {path} 时出错: {e}")
    
    if not video_files:
        print("❌ 未找到任何视频文件")
        return
    
    # 创建输出目录
    os.makedirs("./outputs/results", exist_ok=True)
    
    # 初始化检测器
    detector = GPUMemoryLLaVADetector()
    
    print(f"🎬 开始GPU显存监控处理 {len(video_files)} 个视频...")
    print("=" * 60)
    
    # 处理视频
    results = []
    
    for i, video_file in enumerate(video_files):
        try:
            print(f"\n📹 处理视频 {i+1}/{len(video_files)}: {Path(video_file).name}")
            result = detector.process_video(str(video_file))
            results.append(result)
            
            # 显示显存使用摘要
            if 'memory_stats' in result:
                mem_stats = result['memory_stats']
                print(f"📊 峰值显存: {mem_stats.get('peak_allocated', 0):.1f}MB")
                print(f"📊 显存增长: {mem_stats.get('memory_increase', 0):+.1f}MB")
            
        except Exception as e:
            logger.error(f"❌ 处理视频失败: {e}")
    
    # 保存结果
    print("\n💾 保存GPU显存监控结果...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    memory_result = {
        'metadata': {
            'model': 'CLIP-GPU-Memory-Monitor',
            'device': 'GPU',
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
            'total_gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 if torch.cuda.is_available() else 0,
            'total_videos': len(video_files),
            'timestamp': timestamp,
            'test_type': 'gpu_memory_monitoring'
        },
        'results': results
    }
    
    json_file = f"./outputs/results/gpu_memory_llava_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(memory_result, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("🎉 GPU显存监控完成!")
    print("=" * 60)
    print(f"📊 测试视频数: {len(video_files)}")
    print(f"📄 结果文件: {json_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()