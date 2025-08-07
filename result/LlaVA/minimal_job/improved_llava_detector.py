#!/usr/bin/env python3
"""
改进版LLaVA鬼探头检测器
1. 降低检测阈值至0.15
2. 增加时序分析识别突然出现的物体
"""

import json
import os
import gc
from pathlib import Path
from datetime import datetime
from PIL import Image
import logging
import torch
from typing import List, Dict, Tuple
import numpy as np
import time
import hashlib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedLLaVADetector:
    """改进版LLaVA鬼探头检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processed_videos = []
        
        # 改进的检测阈值 - 从0.3降至0.15
        self.detection_thresholds = {
            'sudden_appearance': 0.15,  # 突然出现检测阈值
            'high_confidence': 0.20,    # 高置信度阈值
            'temporal_pattern': 0.12    # 时序模式阈值
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            logger.info("🔧 正在初始化改进版模型...")
            logger.info(f"🖥️  设备: {self.device}")
            logger.info(f"🎯 新检测阈值: {self.detection_thresholds}")
            
            # 使用CLIP进行图像编码
            from transformers import CLIPProcessor, CLIPModel
            
            logger.info("📥 加载CLIP模型...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # 移到GPU
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
                logger.info(f"✅ 模型已加载到GPU: {torch.cuda.get_device_name()}")
            else:
                logger.warning("⚠️  CUDA不可用，使用CPU")
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    def extract_frames_enhanced(self, video_path: str, num_frames: int = 16) -> List[Image.Image]:
        """增强版帧提取 - 增加帧数以便更好的时序分析"""
        
        logger.info(f"🎬 处理视频: {Path(video_path).name}")
        overall_start = time.time()
        
        # 验证文件
        if not Path(video_path).exists():
            raise ValueError(f"视频文件不存在: {video_path}")
        
        file_size = Path(video_path).stat().st_size
        logger.info(f"📂 文件大小: {file_size / 1024 / 1024:.2f} MB")
        
        try:
            import decord
            from decord import VideoReader
            
            # 设置decord
            decord.bridge.set_bridge('native')
            
            # 读取视频
            video_reader = VideoReader(str(video_path))
            total_frames = len(video_reader)
            
            logger.info(f"📊 总帧数: {total_frames}")
            
            if total_frames == 0:
                raise ValueError(f"视频文件没有帧: {video_path}")
            
            # 获取视频信息
            try:
                fps = video_reader.get_avg_fps()
                duration = total_frames / fps if fps > 0 else 0
                logger.info(f"📊 帧率: {fps:.2f} fps, 时长: {duration:.2f}秒")
            except Exception as e:
                logger.warning(f"⚠️  无法获取视频信息: {e}")
            
            # 均匀分布选择帧
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            logger.info(f"📊 选择{num_frames}帧用于时序分析: {frame_indices[:5].tolist()}...{frame_indices[-3:].tolist()}")
            
            # 提取帧
            frames = []
            for i, idx in enumerate(frame_indices):
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
                
                # 清理临时对象
                del frame, frame_array
            
            overall_time = time.time() - overall_start
            logger.info(f"✅ 提取完成: {len(frames)}帧, 用时{overall_time:.2f}秒")
            
            # 记录处理信息
            self.processed_videos.append({
                'video_path': video_path,
                'file_size_mb': file_size / 1024 / 1024,
                'total_frames': total_frames,
                'extracted_frames': len(frames),
                'extraction_time': overall_time
            })
            
            return frames
            
        except Exception as e:
            overall_time = time.time() - overall_start
            logger.error(f"❌ 帧提取失败 ({overall_time:.2f}秒): {e}")
            raise
    
    def analyze_temporal_patterns(self, image_features: torch.Tensor) -> Dict:
        """分析时序模式 - 识别突然出现的物体"""
        
        logger.info("🕐 开始时序模式分析...")
        
        # 计算连续帧间的特征变化
        frame_changes = []
        for i in range(1, len(image_features)):
            diff = torch.cosine_similarity(
                image_features[i-1].unsqueeze(0),
                image_features[i].unsqueeze(0)
            ).item()
            change = 1 - diff
            frame_changes.append(change)
        
        frame_changes = np.array(frame_changes)
        
        # 计算时序统计特征
        max_change = np.max(frame_changes)
        mean_change = np.mean(frame_changes)
        std_change = np.std(frame_changes)
        
        # 检测突然变化（异常值检测）
        threshold = mean_change + 2 * std_change  # 2个标准差阈值
        sudden_changes = frame_changes > threshold
        sudden_change_count = np.sum(sudden_changes)
        
        # 检测连续变化模式
        # 寻找连续的高变化区域（可能表示物体出现过程）
        high_change_mask = frame_changes > self.detection_thresholds['temporal_pattern']
        continuous_regions = self._find_continuous_regions(high_change_mask)
        
        # 分析变化的梯度（突然性）
        if len(frame_changes) > 1:
            change_gradient = np.gradient(frame_changes)
            max_gradient = np.max(np.abs(change_gradient))
            gradient_peaks = np.where(np.abs(change_gradient) > np.std(change_gradient) * 1.5)[0]
        else:
            max_gradient = 0
            gradient_peaks = []
        
        temporal_analysis = {
            'frame_changes': frame_changes.tolist(),
            'max_change': float(max_change),
            'mean_change': float(mean_change),
            'std_change': float(std_change),
            'sudden_change_threshold': float(threshold),
            'sudden_change_count': int(sudden_change_count),
            'sudden_change_indices': np.where(sudden_changes)[0].tolist(),
            'continuous_change_regions': continuous_regions,
            'max_gradient': float(max_gradient),
            'gradient_peaks': gradient_peaks.tolist(),
            'high_change_frames': np.where(high_change_mask)[0].tolist()
        }
        
        logger.info(f"📊 最大变化: {max_change:.4f}, 平均变化: {mean_change:.4f}")
        logger.info(f"📊 突然变化次数: {sudden_change_count}, 连续变化区域: {len(continuous_regions)}")
        logger.info(f"📊 最大梯度: {max_gradient:.4f}")
        
        return temporal_analysis
    
    def _find_continuous_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """找出连续的True区域"""
        regions = []
        start = None
        
        for i, value in enumerate(mask):
            if value and start is None:
                start = i
            elif not value and start is not None:
                regions.append((start, i - 1))
                start = None
        
        # 处理到结尾的情况
        if start is not None:
            regions.append((start, len(mask) - 1))
        
        return regions
    
    def enhanced_ghost_detection(self, temporal_analysis: Dict) -> Dict:
        """增强版鬼探头检测逻辑"""
        
        max_change = temporal_analysis['max_change']
        mean_change = temporal_analysis['mean_change']
        sudden_change_count = temporal_analysis['sudden_change_count']
        continuous_regions = temporal_analysis['continuous_change_regions']
        max_gradient = temporal_analysis['max_gradient']
        
        # 多维度检测逻辑
        detection_scores = {
            'sudden_appearance': 0.0,
            'temporal_consistency': 0.0,
            'gradient_analysis': 0.0,
            'overall_confidence': 0.0
        }
        
        # 1. 突然出现检测（降低阈值）
        if max_change > self.detection_thresholds['sudden_appearance']:
            detection_scores['sudden_appearance'] = min(1.0, max_change / self.detection_thresholds['high_confidence'])
        
        # 2. 时序一致性检测
        if sudden_change_count > 0:
            # 有突然变化，增加置信度
            detection_scores['temporal_consistency'] = min(1.0, sudden_change_count / 3.0)
        
        # 3. 梯度分析 - 检测急剧变化
        if max_gradient > 0.05:  # 梯度阈值
            detection_scores['gradient_analysis'] = min(1.0, max_gradient / 0.15)
        
        # 4. 连续变化区域分析
        if continuous_regions:
            # 短而集中的变化区域可能表示物体突然出现
            region_lengths = [end - start + 1 for start, end in continuous_regions]
            avg_region_length = np.mean(region_lengths)
            
            # 偏好较短的连续变化（突然出现特征）
            if avg_region_length < 4:  # 短变化区域
                detection_scores['temporal_consistency'] += 0.3
        
        # 综合置信度计算
        detection_scores['overall_confidence'] = (
            detection_scores['sudden_appearance'] * 0.4 +
            detection_scores['temporal_consistency'] * 0.3 +
            detection_scores['gradient_analysis'] * 0.3
        )
        
        # 决策逻辑（降低检测阈值）
        if detection_scores['overall_confidence'] > 0.6:
            ghost_detected = "yes"
            confidence = min(0.95, detection_scores['overall_confidence'])
            ghost_type = "high_confidence"
            risk_level = "high"
        elif detection_scores['overall_confidence'] > 0.4:
            ghost_detected = "yes"
            confidence = detection_scores['overall_confidence']
            ghost_type = "potential"
            risk_level = "medium"
        elif max_change > self.detection_thresholds['temporal_pattern']:
            ghost_detected = "yes"
            confidence = max(0.3, detection_scores['overall_confidence'])
            ghost_type = "low_confidence"
            risk_level = "low"
        else:
            ghost_detected = "no"
            confidence = 0.2
            ghost_type = "none"
            risk_level = "low"
        
        result = {
            "ghost_probing_detected": ghost_detected,
            "confidence": round(confidence, 3),
            "ghost_type": ghost_type,
            "risk_level": risk_level,
            "detection_scores": detection_scores,
            "detection_reasoning": {
                "max_change_vs_threshold": f"{max_change:.4f} vs {self.detection_thresholds['sudden_appearance']}",
                "sudden_changes": sudden_change_count,
                "continuous_regions": len(continuous_regions),
                "max_gradient": max_gradient
            }
        }
        
        logger.info(f"🎯 检测结果: {ghost_detected} (置信度: {confidence:.3f})")
        logger.info(f"🎯 检测类型: {ghost_type}, 风险级别: {risk_level}")
        
        return result
    
    def analyze_frames_enhanced(self, frames: List[Image.Image]) -> Dict:
        """增强版帧分析"""
        if not frames:
            raise ValueError("没有帧可分析")
        
        try:
            logger.info(f"🔍 开始增强分析{len(frames)}帧...")
            analysis_start = time.time()
            
            # 使用CLIP提取图像特征
            inputs = self.clip_processor(images=frames, return_tensors="pt", padding=True)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # 时序模式分析
            temporal_analysis = self.analyze_temporal_patterns(image_features)
            
            # 增强版鬼探头检测
            detection_result = self.enhanced_ghost_detection(temporal_analysis)
            
            analysis_time = time.time() - analysis_start
            logger.info(f"🧠 增强分析完成: {analysis_time:.4f}秒")
            
            # 合并结果
            result = {
                **detection_result,
                "summary": f"增强分析完成，最大帧间变化: {temporal_analysis['max_change']:.4f}",
                "key_actions": f"时序分析{len(frames)}帧，降低阈值检测",
                "emergency_action_needed": "yes" if detection_result["ghost_probing_detected"] == "yes" else "no",
                "temporal_analysis": temporal_analysis,
                "analysis_time": round(analysis_time, 4)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 增强分析失败: {e}")
            raise
    
    def process_video(self, video_path: str) -> Dict:
        """处理单个视频"""
        video_name = Path(video_path).stem
        logger.info(f"🎬 开始处理视频: {video_name}")
        
        start_time = datetime.now()
        
        try:
            # 增强版帧提取（更多帧数用于时序分析）
            frames = self.extract_frames_enhanced(video_path, num_frames=16)
            
            if not frames:
                raise ValueError("无法提取视频帧")
            
            # 增强版分析
            analysis_result = self.analyze_frames_enhanced(frames)
            
            # 计算处理时间
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
                "model": "CLIP-Enhanced-Temporal",
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "frames_analyzed": len(frames),
                "device": self.device,
                "detection_scores": analysis_result.get("detection_scores", {}),
                "detection_reasoning": analysis_result.get("detection_reasoning", {}),
                "temporal_analysis": analysis_result.get("temporal_analysis", {}),
                "analysis_time": analysis_result.get("analysis_time", 0),
                "thresholds_used": self.detection_thresholds
            }
            
            logger.info(f"✅ 处理完成: {video_name} ({processing_time:.2f}s)")
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
    """改进版主函数"""
    print("🚀 改进版LLaVA鬼探头检测 (降低阈值 + 时序分析)")
    print("=" * 70)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
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
    for path in possible_paths:
        try:
            p = Path(path)
            if p.exists():
                found_videos = list(p.glob("**/*.avi"))
                if found_videos:
                    # 处理前10个视频进行测试
                    video_files = found_videos[:10]
                    print(f"✅ 在 {path} 找到 {len(found_videos)} 个视频，处理前 {len(video_files)} 个")
                    break
        except Exception as e:
            print(f"❌ 检查路径 {path} 时出错: {e}")
    
    if not video_files:
        print("❌ 未找到任何视频文件")
        return
    
    # 创建输出目录
    os.makedirs("./outputs/results", exist_ok=True)
    
    # 初始化改进版检测器
    detector = ImprovedLLaVADetector()
    
    print(f"🎬 开始改进版处理 {len(video_files)} 个视频...")
    print("=" * 70)
    
    # 处理视频
    results = []
    failed_count = 0
    
    for i, video_file in enumerate(video_files):
        try:
            print(f"\n📹 处理视频 {i+1}/{len(video_files)}: {Path(video_file).name}")
            result = detector.process_video(str(video_file))
            results.append(result)
            
            if 'error' in result:
                failed_count += 1
                print(f"❌ 处理失败: {result['error']}")
            else:
                ghost_detected = result.get('ghost_probing_label', 'no')
                confidence = result.get('confidence', 0)
                print(f"✅ 检测结果: {ghost_detected} (置信度: {confidence:.3f})")
                
        except Exception as e:
            logger.error(f"❌ 处理视频失败: {e}")
            failed_count += 1
    
    # 保存结果
    print("\n💾 保存改进版检测结果...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    successful_results = [r for r in results if 'error' not in r]
    
    # 统计检测结果
    detected_count = sum(1 for r in successful_results if r.get('ghost_probing_label') == 'yes')
    
    # 保存详细结果
    improved_result = {
        'metadata': {
            'model': 'CLIP-Enhanced-Temporal',
            'device': 'GPU' if torch.cuda.is_available() else 'CPU',
            'total_videos': len(video_files),
            'successful_videos': len(successful_results),
            'failed_videos': failed_count,
            'detected_ghost_probing': detected_count,
            'timestamp': timestamp,
            'improvements': [
                'Lowered detection threshold from 0.3 to 0.15',
                'Added temporal pattern analysis',
                'Increased frame count to 16 for better temporal analysis',
                'Multi-dimensional detection scoring'
            ]
        },
        'processed_videos_details': detector.processed_videos,
        'results': results
    }
    
    json_file = f"./outputs/results/improved_llava_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(improved_result, f, indent=2, ensure_ascii=False)
    
    # 统计
    if successful_results:
        avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        avg_extraction_time = sum(v['extraction_time'] for v in detector.processed_videos) / len(detector.processed_videos)
    else:
        avg_time = avg_extraction_time = 0
    
    print("=" * 70)
    print("🎉 改进版检测完成!")
    print("=" * 70)
    print(f"📊 测试视频数: {len(video_files)}")
    print(f"✅ 成功处理: {len(successful_results)}")
    print(f"🎯 检测到鬼探头: {detected_count}")
    print(f"❌ 处理失败: {failed_count}")
    if successful_results:
        print(f"⏱️  平均总时间: {avg_time:.2f}秒/视频")
        print(f"⏱️  平均抽帧时间: {avg_extraction_time:.2f}秒/视频")
    print(f"📄 结果文件: {json_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()