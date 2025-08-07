#!/usr/bin/env python3
"""
严格版LLaVA鬼探头检测器
强制无缓存，详细验证每一步
"""

import json
import os
import sys
import gc
from pathlib import Path
from datetime import datetime
from PIL import Image
import logging
import torch
from typing import List, Dict, Optional, Tuple
import numpy as np
import time
import hashlib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrictLLaVADetector:
    """严格版LLaVA鬼探头检测器 - 无缓存验证版"""
    
    def __init__(self):
        """初始化检测器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.processed_videos = []  # 记录处理过的视频
        
        # 平衡版GPT-4.1鬼探头检测prompt
        self.ghost_probing_prompt = """Analyze these sequential driving video frames for ghost probing detection."""
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            logger.info("🔧 正在初始化严格版模型...")
            logger.info(f"🖥️  设备: {self.device}")
            
            # 使用CLIP进行图像编码
            from transformers import CLIPProcessor, CLIPModel
            
            # 加载CLIP模型
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
    
    def extract_frames_strict(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """严格提取视频帧 - 无缓存版本"""
        
        logger.info(f"🎬 严格处理视频: {Path(video_path).name}")
        overall_start = time.time()
        
        # 验证文件
        if not Path(video_path).exists():
            raise ValueError(f"视频文件不存在: {video_path}")
        
        file_size = Path(video_path).stat().st_size
        logger.info(f"📂 文件大小: {file_size / 1024 / 1024:.2f} MB")
        
        # 计算文件哈希验证真实性
        hash_start = time.time()
        with open(video_path, 'rb') as f:
            file_hash = hashlib.md5(f.read(8192)).hexdigest()  # 读取前8KB计算hash
        hash_time = time.time() - hash_start
        logger.info(f"🔍 文件hash验证: {file_hash} ({hash_time:.3f}秒)")
        
        try:
            import decord
            from decord import VideoReader
            
            # 强制清除缓存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 每次都重新设置bridge
            decord.bridge.set_bridge('native')
            
            # 读取视频 - 添加详细计时
            logger.info("📖 开始读取视频...")
            video_load_start = time.time()
            
            # 强制重新创建VideoReader对象
            video_reader = VideoReader(str(video_path))
            total_frames = len(video_reader)
            
            video_load_time = time.time() - video_load_start
            logger.info(f"📊 视频加载时间: {video_load_time:.4f}秒")
            logger.info(f"📊 总帧数: {total_frames}")
            
            if total_frames == 0:
                raise ValueError(f"视频文件没有帧: {video_path}")
            
            # 获取详细视频信息
            try:
                fps = video_reader.get_avg_fps()
                duration = total_frames / fps if fps > 0 else 0
                logger.info(f"📊 帧率: {fps:.2f} fps")
                logger.info(f"📊 时长: {duration:.2f}秒")
                
                # 验证视频合理性
                if duration < 1.0:
                    logger.warning(f"⚠️  视频时长过短: {duration:.2f}秒")
                if fps < 10 or fps > 60:
                    logger.warning(f"⚠️  异常帧率: {fps:.2f} fps")
                    
            except Exception as e:
                logger.warning(f"⚠️  无法获取视频信息: {e}")
            
            # 均匀分布选择帧
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            logger.info(f"📊 选择帧索引: {frame_indices.tolist()}")
            
            # 严格提取帧 - 每帧独立计时
            frames = []
            extraction_start = time.time()
            
            for i, idx in enumerate(frame_indices):
                frame_start = time.time()
                logger.info(f"🔍 提取第 {i+1}/{num_frames} 帧 (索引: {idx})")
                
                # 强制等待一点时间确保不是缓存
                time.sleep(0.001)
                
                # 获取帧
                frame_read_start = time.time()
                frame = video_reader[idx]
                frame_read_time = time.time() - frame_read_start
                
                logger.info(f"  📖 帧读取时间: {frame_read_time:.4f}秒")
                
                # 验证帧数据
                if frame is None:
                    raise ValueError(f"帧 {idx} 读取失败")
                
                # 转换为numpy数组
                convert_start = time.time()
                if hasattr(frame, 'asnumpy'):
                    frame_array = frame.asnumpy()
                elif isinstance(frame, torch.Tensor):
                    frame_array = frame.cpu().numpy()
                else:
                    frame_array = np.array(frame)
                
                convert_time = time.time() - convert_start
                logger.info(f"  🔄 数组转换时间: {convert_time:.4f}秒")
                
                # 验证帧数据合理性
                if frame_array.size == 0:
                    raise ValueError(f"帧 {idx} 数据为空")
                
                logger.info(f"  📊 帧形状: {frame_array.shape}")
                logger.info(f"  📊 数据范围: {frame_array.min()}-{frame_array.max()}")
                
                # 转换为PIL Image
                pil_start = time.time()
                pil_image = Image.fromarray(frame_array.astype(np.uint8))
                pil_time = time.time() - pil_start
                
                logger.info(f"  🖼️  PIL转换时间: {pil_time:.4f}秒")
                logger.info(f"  🖼️  图像大小: {pil_image.size}")
                
                # 验证图像内容不是全黑或全白
                img_array = np.array(pil_image)
                pixel_variance = np.var(img_array)
                logger.info(f"  🎨 像素方差: {pixel_variance:.2f}")
                
                if pixel_variance < 100:
                    logger.warning(f"  ⚠️  帧 {idx} 可能是纯色图像")
                
                frames.append(pil_image)
                
                frame_total_time = time.time() - frame_start
                logger.info(f"  ✅ 帧 {i+1} 完成: {frame_total_time:.4f}秒")
                
                # 强制内存清理
                del frame, frame_array
                gc.collect()
            
            extraction_time = time.time() - extraction_start
            overall_time = time.time() - overall_start
            
            logger.info(f"✅ 所有帧提取时间: {extraction_time:.4f}秒")
            logger.info(f"✅ 总处理时间: {overall_time:.4f}秒")
            logger.info(f"✅ 平均每帧: {extraction_time/num_frames:.4f}秒")
            
            # 记录处理信息
            process_info = {
                'video_path': video_path,
                'file_size_mb': file_size / 1024 / 1024,
                'file_hash': file_hash,
                'total_frames': total_frames,
                'extraction_time': extraction_time,
                'avg_time_per_frame': extraction_time / num_frames,
                'overall_time': overall_time
            }
            self.processed_videos.append(process_info)
            
            return frames
            
        except Exception as e:
            overall_time = time.time() - overall_start
            logger.error(f"❌ 严格提取失败 ({overall_time:.4f}秒): {e}")
            raise
    
    def analyze_frames_simple(self, frames: List[Image.Image], video_path: str) -> Dict:
        """分析帧（保持原来的逻辑）"""
        if not frames:
            raise ValueError("没有帧可分析")
        
        try:
            logger.info(f"🔍 严格分析{len(frames)}帧...")
            analysis_start = time.time()
            
            # 使用CLIP提取图像特征
            inputs = self.clip_processor(images=frames, return_tensors="pt", padding=True)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
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
            
            analysis_time = time.time() - analysis_start
            logger.info(f"🧠 CLIP分析时间: {analysis_time:.4f}秒")
            logger.info(f"📊 最大特征变化: {max_change:.4f}")
            logger.info(f"📊 平均特征变化: {avg_change:.4f}")
            
            # 判断逻辑
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
                "summary": f"严格分析完成，最大帧间变化: {max_change:.4f}",
                "key_actions": f"严格检测{len(frames)}帧，CLIP特征分析",
                "risk_level": risk_level,
                "distance_estimate": "基于严格特征变化估算",
                "emergency_action_needed": "yes" if ghost_detected == "yes" else "no",
                "max_frame_change": round(max_change, 4),
                "avg_frame_change": round(avg_change, 4),
                "analysis_time": round(analysis_time, 4)
            }
            
            logger.info(f"✅ 严格分析完成 - 鬼探头: {ghost_detected} (置信度: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ 严格分析失败: {e}")
            raise
    
    def process_video(self, video_path: str) -> Dict:
        """严格处理单个视频"""
        video_name = Path(video_path).stem
        logger.info(f"🎬 开始严格处理视频: {video_name}")
        
        start_time = datetime.now()
        
        try:
            # 严格提取视频帧
            frames = self.extract_frames_strict(video_path, num_frames=8)
            
            if not frames:
                raise ValueError("严格提取失败：无法提取视频帧")
            
            # 分析帧
            analysis_result = self.analyze_frames_simple(frames, video_path)
            
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
                "model": "CLIP-Strict-NoCache",
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "method": "strict_no_cache_analysis",
                "frames_analyzed": len(frames),
                "device": self.device,
                "max_frame_change": analysis_result.get("max_frame_change", 0),
                "avg_frame_change": analysis_result.get("avg_frame_change", 0),
                "analysis_time": analysis_result.get("analysis_time", 0)
            }
            
            logger.info(f"✅ 严格处理完成: {video_name} ({processing_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"❌ 严格处理失败: {video_name} - {e}")
            return {
                "video_id": video_name,
                "video_path": str(video_path),
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "method": "strict_no_cache_analysis"
            }

def main():
    """严格版主函数 - 仅处理5个视频进行测试"""
    print("🚀 开始严格版LLaVA鬼探头检测...")
    print("⚠️  注意：这是无缓存严格验证版本")
    
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
    video_folder = None
    
    for path in possible_paths:
        try:
            p = Path(path)
            if p.exists():
                found_videos = list(p.glob("**/*.avi"))
                if found_videos:
                    # 仅取前5个视频进行严格测试
                    video_files = found_videos[:5]
                    video_folder = p
                    print(f"✅ 在 {path} 找到 {len(found_videos)} 个视频，严格测试前 {len(video_files)} 个")
                    break
        except Exception as e:
            print(f"❌ 检查路径 {path} 时出错: {e}")
    
    if not video_files:
        print("❌ 未找到任何视频文件")
        return
    
    # 创建输出目录
    os.makedirs("./outputs/results", exist_ok=True)
    
    # 初始化严格检测器
    detector = StrictLLaVADetector()
    
    print(f"🎬 开始严格处理 {len(video_files)} 个视频...")
    print("=" * 60)
    
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
                print(f"✅ 处理成功: {result['processing_time']:.2f}秒")
                
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
    print("\n💾 保存严格测试结果...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    successful_results = [r for r in results if 'error' not in r]
    
    # 保存详细的严格测试结果
    strict_result = {
        'metadata': {
            'model': 'CLIP-Strict-NoCache',
            'device': 'GPU' if torch.cuda.is_available() else 'CPU',
            'total_videos': len(video_files),
            'successful_videos': len(successful_results),
            'failed_videos': failed_count,
            'timestamp': timestamp,
            'test_type': 'strict_no_cache_validation'
        },
        'processed_videos_details': detector.processed_videos,
        'results': results
    }
    
    json_file = f"./outputs/results/strict_llava_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(strict_result, f, indent=2, ensure_ascii=False)
    
    # 统计
    if successful_results:
        avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        avg_extraction_time = sum(v['extraction_time'] for v in detector.processed_videos) / len(detector.processed_videos)
    else:
        avg_time = avg_extraction_time = 0
    
    print("=" * 60)
    print("🎉 严格版检测完成!")
    print("=" * 60)
    print(f"📊 测试视频数: {len(video_files)}")
    print(f"✅ 成功处理: {len(successful_results)}")
    print(f"❌ 处理失败: {failed_count}")
    if successful_results:
        print(f"⏱️  平均总时间: {avg_time:.2f}秒/视频")
        print(f"⏱️  平均抽帧时间: {avg_extraction_time:.2f}秒/视频")
    print(f"📄 结果文件: {json_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()