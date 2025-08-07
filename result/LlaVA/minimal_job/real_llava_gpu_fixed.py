#!/usr/bin/env python3
"""
真正的LLaVA鬼探头检测器 - GPU版本修复
修复decord返回tensor的问题，保持GPU加速
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

class RealLLaVADetector:
    """真实的LLaVA鬼探头检测器 - GPU加速版"""
    
    def __init__(self, model_path: str = "lmms-lab/LLaVA-NeXT-Video-7B-DPO"):
        """
        初始化LLaVA检测器
        
        Args:
            model_path: LLaVA模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 平衡版GPT-4.1鬼探头检测prompt
        self.ghost_probing_prompt = """You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a driving video. Focus on changes in relative positions, distances, and speeds of objects, particularly vehicles and pedestrians, and how these might indicate potential collision risks.

IMPORTANT: For ghost probing detection, consider TWO categories:

**1. HIGH-CONFIDENCE Ghost Probing (use "ghost probing" in analysis)**:
- Object appears EXTREMELY close (within 1-2 vehicle lengths, <3 meters) 
- Appearance is SUDDEN and from blind spots (behind parked cars, buildings, corners)
- Occurs in HIGH-RISK environments: highways, rural roads, parking lots, uncontrolled intersections
- Requires IMMEDIATE emergency braking/swerving to avoid collision
- Movement is COMPLETELY UNPREDICTABLE and violates traffic expectations

**2. POTENTIAL Ghost Probing (use "potential ghost probing" in analysis)**:
- Object appears suddenly but at moderate distance (3-5 meters)
- Sudden movement in environments where some unpredictability exists
- Requires emergency braking but collision risk is moderate
- Movement is unexpected but not completely impossible given the context

**3. NORMAL Traffic Situations (do NOT use "ghost probing")**:
- Pedestrians crossing at intersections, crosswalks, or traffic lights
- Vehicles making normal lane changes, turns, or merging with signals
- Cyclists following predictable paths in urban areas or bike lanes
- Any movement that is EXPECTED given the traffic environment and context

Analyze these sequential frames and respond with a JSON object:
{
    "ghost_probing_detected": "yes/no",
    "confidence": 0.0-1.0,
    "ghost_type": "high_confidence/potential/none",
    "summary": "brief description of the scene",
    "key_actions": "description of most important actions",
    "risk_level": "high/medium/low",
    "distance_estimate": "distance to closest threat in meters",
    "emergency_action_needed": "yes/no"
}"""
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化LLaVA模型 - 使用GPU"""
        try:
            logger.info("🔧 正在初始化LLaVA模型...")
            logger.info(f"🖥️  设备: {self.device}")
            
            # 导入LLaVA相关模块
            from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration
            
            # 加载模型和处理器 - 使用GPU
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",  # 自动分配到GPU
                low_cpu_mem_usage=True
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            # 确保模型在GPU上
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info(f"✅ LLaVA模型已加载到GPU: {torch.cuda.get_device_name()}")
            else:
                logger.warning("⚠️  CUDA不可用，使用CPU")
            
        except Exception as e:
            logger.error(f"❌ LLaVA模型初始化失败: {e}")
            raise
    
    def extract_frames_with_decord(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """
        使用decord库提取视频帧 - 修复tensor问题
        
        Args:
            video_path: 视频文件路径
            num_frames: 提取帧数
            
        Returns:
            PIL Image列表
        """
        try:
            import decord
            from decord import VideoReader
            
            # 设置decord使用native bridge (返回numpy数组)
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
                # 获取帧 - 现在应该返回numpy数组
                frame = video_reader[idx]
                
                # 如果还是tensor，转换为numpy
                if hasattr(frame, 'asnumpy'):
                    frame_array = frame.asnumpy()
                elif isinstance(frame, torch.Tensor):
                    frame_array = frame.cpu().numpy()
                else:
                    frame_array = np.array(frame)
                
                # 转换为PIL Image
                pil_image = Image.fromarray(frame_array.astype(np.uint8))
                frames.append(pil_image)
            
            logger.info(f"✅ 使用decord从视频中提取了{len(frames)}帧")
            return frames
            
        except Exception as e:
            logger.error(f"❌ 视频帧提取失败: {e}")
            raise
    
    def analyze_frames_with_llava(self, frames: List[Image.Image], video_path: str) -> Dict:
        """
        使用LLaVA模型分析视频帧 - GPU加速
        
        Args:
            frames: PIL Image帧列表
            video_path: 视频路径（用于错误报告）
            
        Returns:
            分析结果字典
        """
        if self.model is None or not frames:
            raise ValueError("LLaVA模型未初始化或没有帧可分析")
        
        try:
            logger.info(f"🔍 正在使用LLaVA分析{len(frames)}帧...")
            
            # 准备输入
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.ghost_probing_prompt},
                        *[{"type": "image"} for _ in frames]
                    ],
                }
            ]
            
            # 处理输入
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(
                text=prompt, 
                images=frames, 
                return_tensors="pt"
            )
            
            # 将输入移到GPU
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # 生成响应 - 在GPU上进行推理
            with torch.no_grad():
                with torch.cuda.amp.autocast():  # 使用混合精度加速
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        use_cache=True
                    )
            
            # 解码输出
            generated_ids = output[0][inputs['input_ids'].shape[1]:]
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            logger.info(f"🤖 LLaVA响应: {generated_text[:200]}...")
            
            # 尝试解析JSON响应
            try:
                # 寻找JSON部分
                json_start = generated_text.find('{')
                json_end = generated_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = generated_text[json_start:json_end]
                    result = json.loads(json_str)
                    logger.info("✅ 成功解析LLaVA的JSON响应")
                    return result
                else:
                    return self._parse_text_response(generated_text)
            except json.JSONDecodeError:
                logger.warning("⚠️  LLaVA响应不是有效JSON，使用文本解析")
                return self._parse_text_response(generated_text)
                
        except Exception as e:
            logger.error(f"❌ LLaVA分析失败: {e}")
            raise
    
    def _parse_text_response(self, text: str) -> Dict:
        """解析文本响应为结构化数据"""
        text_lower = text.lower()
        
        # 检测鬼探头关键词
        ghost_detected = any(keyword in text_lower for keyword in [
            'ghost probing', 'sudden appearance', 'emergency braking', 
            'collision risk', 'immediate threat', 'extremely close'
        ])
        
        # 检测高确信度指标
        high_confidence = any(keyword in text_lower for keyword in [
            'extremely close', 'immediate', 'emergency', '<3 meters', 'sudden'
        ])
        
        # 估算置信度
        if ghost_detected and high_confidence:
            confidence = 0.85
            ghost_type = 'high_confidence'
            risk_level = 'high'
        elif ghost_detected:
            confidence = 0.65
            ghost_type = 'potential'
            risk_level = 'medium'
        else:
            confidence = 0.25
            ghost_type = 'none'
            risk_level = 'low'
        
        return {
            "ghost_probing_detected": "yes" if ghost_detected else "no",
            "confidence": confidence,
            "ghost_type": ghost_type,
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "key_actions": "LLaVA文本分析结果",
            "risk_level": risk_level,
            "distance_estimate": "分析中确定",
            "emergency_action_needed": "yes" if ghost_detected else "no"
        }
    
    def process_video(self, video_path: str) -> Dict:
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            检测结果
        """
        video_name = Path(video_path).stem
        logger.info(f"🎬 开始处理视频: {video_name}")
        
        start_time = datetime.now()
        
        try:
            # 提取真实视频帧
            frames = self.extract_frames_with_decord(video_path, num_frames=8)
            
            if not frames:
                raise ValueError("无法提取视频帧")
            
            # 使用LLaVA分析 - GPU加速
            analysis_result = self.analyze_frames_with_llava(frames, video_path)
            
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
                "model": "LLaVA-NeXT-Video-7B-DPO",
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "method": "real_llava_gpu_analysis",
                "frames_analyzed": len(frames),
                "device": self.device
            }
            
            logger.info(f"✅ 视频处理完成: {video_name} ({processing_time:.1f}s) - 鬼探头: {result['ghost_probing_label']}")
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
    """主函数 - 处理100个DADA视频"""
    print("🚀 开始真正的LLaVA鬼探头检测（GPU加速版）...")
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
        print(f"📊 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  GPU不可用，将使用CPU")
    
    # 获取视频数据路径
    azureml_data_path = os.environ.get('AZUREML_DATAREFERENCE_video_data')
    
    possible_paths = []
    if azureml_data_path:
        possible_paths.append(azureml_data_path)
        print(f"🔧 从环境变量找到数据路径: {azureml_data_path}")
    
    possible_paths.extend([
        "./inputs/video_data", 
        "./inputs",
        "."
    ])
    
    video_files = []
    video_folder = None
    
    for path in possible_paths:
        try:
            p = Path(path)
            if p.exists():
                found_videos = list(p.glob("**/*.avi"))
                if found_videos:
                    video_files = found_videos[:100]  # 限制100个
                    video_folder = p
                    print(f"✅ 在 {path} 找到 {len(video_files)} 个视频文件")
                    break
                else:
                    print(f"⚠️  路径 {path} 存在但没有.avi文件")
        except Exception as e:
            print(f"❌ 检查路径 {path} 时出错: {e}")
    
    if not video_files:
        print("❌ 未找到任何视频文件，无法进行真实分析")
        return
    
    # 创建输出目录
    os.makedirs("./outputs/results", exist_ok=True)
    
    # 初始化检测器
    detector = RealLLaVADetector()
    
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
                print(f"📊 处理进度: {i+1}/{len(video_files)} ({(i+1)/len(video_files)*100:.1f}%) - 成功: {success_count}, 失败: {failed_count}")
                
        except Exception as e:
            logger.error(f"❌ 处理视频 {video_file} 失败: {e}")
            failed_count += 1
            results.append({
                "video_id": Path(video_file).stem,
                "video_path": str(video_file),
                "error": str(e),
                "processing_time": 0
            })
    
    print("💾 保存结果文件...")
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 过滤成功的结果
    successful_results = [r for r in results if 'error' not in r]
    
    # JSON格式结果
    json_file = f"./outputs/results/real_llava_gpu_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'model': 'LLaVA-NeXT-Video-7B-DPO',
                'device': 'GPU' if torch.cuda.is_available() else 'CPU',
                'prompt_version': 'balanced_gpt41_compatible',
                'total_videos': len(video_files),
                'successful_videos': len(successful_results),
                'failed_videos': failed_count,
                'timestamp': timestamp,
                'video_folder': str(video_folder) if video_folder else 'not_found'
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    # CSV格式结果（仅成功的结果）
    if successful_results:
        csv_file = f"./outputs/results/real_llava_gpu_results_{timestamp}.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write('video_id,ghost_probing_label,confidence,ghost_type,risk_level,processing_time,method\n')
            for r in successful_results:
                f.write(f"{r['video_id']},{r['ghost_probing_label']},{r['confidence']:.3f},{r['ghost_type']},{r['risk_level']},{r['processing_time']:.1f},{r['method']}\n")
    
    # 统计信息
    if successful_results:
        ghost_count = len([r for r in successful_results if r['ghost_probing_label'] == 'yes'])
        normal_count = len(successful_results) - ghost_count
        detection_rate = (ghost_count / len(successful_results)) * 100
        avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
    else:
        ghost_count = normal_count = detection_rate = avg_processing_time = 0
    
    summary = {
        'total_videos': len(video_files),
        'successful_processed': len(successful_results),
        'failed_processed': failed_count,
        'ghost_probing_detected': ghost_count,
        'normal_videos': normal_count,
        'detection_rate_percent': round(detection_rate, 2),
        'average_processing_time': round(avg_processing_time, 2),
        'timestamp': timestamp,
        'device': 'GPU' if torch.cuda.is_available() else 'CPU',
        'files_generated': [json_file] + ([csv_file] if successful_results else [])
    }
    
    summary_file = f"./outputs/results/real_llava_gpu_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("🎉 真正的LLaVA鬼探头检测完成 (GPU加速)!")
    print("=" * 60)
    print(f"📊 总视频数: {len(video_files)}")
    print(f"✅ 成功处理: {len(successful_results)}")
    print(f"❌ 处理失败: {failed_count}")
    if successful_results:
        print(f"🚨 鬼探头检测: {ghost_count} ({detection_rate:.1f}%)")
        print(f"📈 正常视频: {normal_count} ({100-detection_rate:.1f}%)")
        print(f"⏱️  平均处理时间: {avg_processing_time:.1f}秒/视频")
        print(f"📊 CSV文件: {csv_file}")
    print(f"📄 结果文件: {json_file}")
    print(f"📋 统计文件: {summary_file}")
    print(f"🖥️  设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

if __name__ == "__main__":
    main()