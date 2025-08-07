#!/usr/bin/env python3
"""
WiseAD A100 GPU 鬼探头检测系统 v2.0
改进版：确保输出保存、10秒段检测、详细日志记录
模拟GPT-4.1 Balanced的检测格式和输出
"""

import os
import sys
import json
import cv2
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/wisead_ghost_detailed.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """安装WiseAD依赖"""
    logger.info("🔧 开始安装WiseAD A100 GPU依赖...")
    
    dependencies = [
        "ultralytics>=8.0.0",
        "opencv-python-headless>=4.5.0", 
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "azure-storage-blob>=12.0.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0"
    ]
    
    for dep in dependencies:
        logger.info(f"📦 安装 {dep}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ {dep} 安装成功")
        else:
            logger.error(f"❌ {dep} 安装失败: {result.stderr}")
            return False
    
    return True

def verify_environment():
    """验证运行环境"""
    try:
        import cv2
        logger.info("✅ OpenCV 验证成功")
        
        import torch
        logger.info(f"✅ PyTorch 验证成功: {torch.__version__}")
        
        from ultralytics import YOLO
        logger.info("✅ YOLO 验证成功")
        
        from azure.storage.blob import BlobServiceClient
        logger.info("✅ Azure Storage 验证成功")
        
        return True
    except ImportError as e:
        logger.error(f"❌ 环境验证失败: {e}")
        return False

def setup_wisead_model():
    """设置WiseAD模型"""
    try:
        from ultralytics import YOLO
        import torch
        
        # 检查GPU
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"🚀 使用A100 GPU: {device} ({memory:.1f}GB)")
        else:
            logger.warning("⚠️ 未检测到GPU，使用CPU")
        
        # 加载YOLO模型
        logger.info("🤖 加载WiseAD YOLO模型...")
        model = YOLO('yolov8s.pt')
        
        # 移动到GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        logger.info(f"🎯 WiseAD模型已转移到: {device}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ WiseAD模型设置失败: {e}")
        return None

def download_dada_videos():
    """从Azure Storage下载DADA视频"""
    try:
        from azure.storage.blob import BlobServiceClient
        
        # Azure Storage配置
        account_name = "drivelmmstorage2e932dad7"
        account_key = "YQDfQLwbRfF8bpGx2YBaxm2VGN8zPHKqrYPeq/Y+gGo7+7kbC60+nfgJlv7a3NqQIGRKp4DGmOmz+AStREIxgA=="
        container_name = "dada-videos"
        
        blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=account_key
        )
        
        logger.info("📥 从Azure Storage下载DADA视频: dada-videos")
        
        container_client = blob_service_client.get_container_client(container_name)
        blobs = list(container_client.list_blobs())
        
        # 过滤出需要的视频文件
        video_files = [blob.name for blob in blobs if blob.name.endswith('.avi')]
        video_files = [f for f in video_files if any(f.startswith(f"images_{i}_") for i in range(1, 6))]
        video_files.sort()
        
        logger.info(f"📹 找到 {len(video_files)} 个DADA视频待下载")
        
        # 创建本地目录
        local_dir = "/tmp/dada_videos"
        os.makedirs(local_dir, exist_ok=True)
        
        downloaded_videos = []
        for i, video_file in enumerate(video_files, 1):
            local_path = os.path.join(local_dir, video_file)
            
            if os.path.exists(local_path):
                logger.info(f"📥 跳过已存在视频 {i}/{len(video_files)}: {video_file}")
                downloaded_videos.append(local_path)
                continue
                
            logger.info(f"📥 下载视频 {i}/{len(video_files)}: {video_file}")
            
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=video_file)
            
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            logger.info(f"✅ 下载成功: {video_file}")
            downloaded_videos.append(local_path)
        
        logger.info(f"📥 DADA视频下载完成: {len(downloaded_videos)}/{len(video_files)}")
        return downloaded_videos
        
    except Exception as e:
        logger.error(f"❌ 视频下载失败: {e}")
        return []

def detect_ghost_probing_10s_segments(model, video_path):
    """
    WiseAD 10秒段鬼探头检测
    模拟GPT-4.1 Balanced的检测格式
    """
    video_id = os.path.basename(video_path).replace('.avi', '')
    logger.info(f"👻 开始WiseAD鬼探头检测 (10秒段模式): {video_id}")
    
    try:
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ 无法打开视频: {video_path}")
            return None
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"📹 视频信息: {total_frames}帧, {fps:.1f}FPS, {duration:.1f}秒")
        
        # 10秒段检测配置
        segment_duration = 10  # 10秒一段
        frames_per_segment = int(fps * segment_duration)
        total_segments = int(duration / segment_duration) + (1 if duration % segment_duration > 0 else 0)
        
        ghost_results = {
            "video_id": video_id,
            "video_info": {
                "duration": duration,
                "fps": fps,
                "total_frames": total_frames,
                "total_segments": total_segments
            },
            "segments": [],
            "ghost_summary": {
                "total_ghost_events": 0,
                "high_risk_events": 0,
                "potential_events": 0,
                "segments_with_ghosts": 0
            },
            "processing_time": {
                "start_time": datetime.now().isoformat(),
                "end_time": None
            }
        }
        
        logger.info(f"🎬 开始处理 {total_segments} 个10秒段")
        
        # 逐段检测
        for segment_idx in range(total_segments):
            start_frame = segment_idx * frames_per_segment
            end_frame = min(start_frame + frames_per_segment, total_frames)
            start_time_sec = start_frame / fps
            end_time_sec = end_frame / fps
            
            logger.info(f"🔍 处理段 {segment_idx + 1}/{total_segments}: {start_time_sec:.1f}s - {end_time_sec:.1f}s")
            
            segment_result = {
                "segment_id": segment_idx + 1,
                "time_range": {
                    "start": start_time_sec,
                    "end": end_time_sec
                },
                "frame_range": {
                    "start": start_frame,
                    "end": end_frame
                },
                "ghost_events": [],
                "segment_summary": {
                    "ghost_count": 0,
                    "high_risk": 0,
                    "potential": 0
                }
            }
            
            # 设置视频位置到段开始
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 在该段中每隔1秒检测一帧
            detection_interval = int(fps)  # 每秒检测一帧
            frame_detections = []
            
            for frame_offset in range(0, frames_per_segment, detection_interval):
                current_frame = start_frame + frame_offset
                if current_frame >= end_frame:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # YOLO检测
                results = model(frame, verbose=False)
                
                # 分析检测结果寻找鬼探头
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # 提取检测信息
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            bbox = box.xyxy[0].tolist()
                            
                            # 鬼探头判定逻辑
                            if class_name in ['person', 'bicycle', 'motorcycle'] and confidence > 0.5:
                                # 基于位置和置信度评估鬼探头风险
                                ghost_score = calculate_ghost_risk(bbox, frame.shape, confidence, class_name)
                                
                                if ghost_score > 0.3:  # 鬼探头阈值
                                    current_time = current_frame / fps
                                    
                                    ghost_event = {
                                        "frame": current_frame,
                                        "time": current_time,
                                        "object_class": class_name,
                                        "confidence": confidence,
                                        "bbox": bbox,
                                        "ghost_score": ghost_score,
                                        "risk_level": "high" if ghost_score > 0.7 else "potential"
                                    }
                                    
                                    segment_result["ghost_events"].append(ghost_event)
                                    segment_result["segment_summary"]["ghost_count"] += 1
                                    
                                    if ghost_score > 0.7:
                                        segment_result["segment_summary"]["high_risk"] += 1
                                        logger.info(f"🔥 检测到高风险鬼探头: 帧{current_frame}, {class_name}, 风险:{ghost_score:.3f}")
                                    else:
                                        segment_result["segment_summary"]["potential"] += 1
                                        logger.info(f"⚠️ 检测到潜在鬼探头: 帧{current_frame}, {class_name}, 风险:{ghost_score:.3f}")
            
            # 添加段结果
            ghost_results["segments"].append(segment_result)
            
            # 更新总体统计
            ghost_results["ghost_summary"]["total_ghost_events"] += segment_result["segment_summary"]["ghost_count"]
            ghost_results["ghost_summary"]["high_risk_events"] += segment_result["segment_summary"]["high_risk"]
            ghost_results["ghost_summary"]["potential_events"] += segment_result["segment_summary"]["potential"]
            
            if segment_result["segment_summary"]["ghost_count"] > 0:
                ghost_results["ghost_summary"]["segments_with_ghosts"] += 1
        
        cap.release()
        
        # 完成处理
        ghost_results["processing_time"]["end_time"] = datetime.now().isoformat()
        
        total_events = ghost_results["ghost_summary"]["total_ghost_events"]
        high_risk = ghost_results["ghost_summary"]["high_risk_events"]
        potential = ghost_results["ghost_summary"]["potential_events"]
        
        logger.info(f"✅ WiseAD鬼探头检测完成: {video_id}")
        logger.info(f"👻 总鬼探头事件: {total_events}")
        logger.info(f"🔥 高风险事件: {high_risk}")
        logger.info(f"⚠️ 潜在风险事件: {potential}")
        logger.info(f"📊 有鬼探头的段数: {ghost_results['ghost_summary']['segments_with_ghosts']}/{total_segments}")
        
        return ghost_results
        
    except Exception as e:
        logger.error(f"❌ 鬼探头检测失败: {e}")
        return None

def calculate_ghost_risk(bbox, frame_shape, confidence, class_name):
    """计算鬼探头风险评分"""
    # 基础评分
    base_score = confidence * 0.3
    
    # 位置评分 (边缘出现更危险)
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    h, w = frame_shape[:2]
    
    # 边缘距离评分
    edge_distance = min(center_x, w - center_x, center_y, h - center_y)
    edge_score = max(0, (50 - edge_distance) / 50) * 0.4
    
    # 目标大小评分 (小目标更可能是鬼探头)
    obj_area = (x2 - x1) * (y2 - y1)
    frame_area = w * h
    size_ratio = obj_area / frame_area
    size_score = max(0, (0.1 - size_ratio) / 0.1) * 0.3
    
    # 类别评分
    class_scores = {
        'person': 0.8,
        'bicycle': 0.6,
        'motorcycle': 0.7,
        'car': 0.2
    }
    class_score = class_scores.get(class_name, 0.1)
    
    total_score = base_score + edge_score + size_score * class_score
    return min(total_score, 1.0)

def save_ghost_results(ghost_results, output_dir="/tmp/wisead_results"):
    """保存鬼探头检测结果"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        video_id = ghost_results["video_id"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        detail_file = os.path.join(output_dir, f"wisead_ghost_{video_id}_detailed.json")
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(ghost_results, f, indent=2, ensure_ascii=False)
        
        # 保存简化结果 (类似GPT-4.1格式)
        simple_result = {
            "video_id": video_id,
            "ghost_probing_analysis": {
                "total_segments": len(ghost_results["segments"]),
                "segments_with_ghosts": ghost_results["ghost_summary"]["segments_with_ghosts"],
                "total_ghost_events": ghost_results["ghost_summary"]["total_ghost_events"],
                "high_risk_events": ghost_results["ghost_summary"]["high_risk_events"],
                "potential_events": ghost_results["ghost_summary"]["potential_events"]
            },
            "risk_assessment": "HIGH" if ghost_results["ghost_summary"]["high_risk_events"] > 5 else 
                              "MEDIUM" if ghost_results["ghost_summary"]["total_ghost_events"] > 2 else "LOW"
        }
        
        simple_file = os.path.join(output_dir, f"wisead_ghost_{video_id}_summary.json")
        with open(simple_file, 'w', encoding='utf-8') as f:
            json.dump(simple_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 WiseAD结果已保存:")
        logger.info(f"   - 详细结果: {detail_file}")
        logger.info(f"   - 简化结果: {simple_file}")
        
        return detail_file, simple_file
        
    except Exception as e:
        logger.error(f"❌ 结果保存失败: {e}")
        return None, None

def main():
    """主函数"""
    logger.info("🚀 启动WiseAD A100 鬼探头检测系统 v2.0")
    logger.info("🎯 10秒段检测模式，详细日志记录")
    
    # 1. 安装依赖
    if not install_dependencies():
        logger.error("❌ 依赖安装失败")
        return
    
    logger.info("✅ 所有WiseAD依赖安装验证完成")
    
    # 2. 验证环境
    if not verify_environment():
        logger.error("❌ 环境验证失败") 
        return
    
    logger.info("✅ WiseAD核心模块导入成功")
    
    # 3. 设置模型
    model = setup_wisead_model()
    if model is None:
        logger.error("❌ WiseAD模型设置失败")
        return
    
    # 4. 下载视频
    video_files = download_dada_videos()
    if not video_files:
        logger.error("❌ 视频下载失败")
        return
    
    # 5. 批量处理视频
    logger.info(f"🎬 准备使用WiseAD分析 {len(video_files)} 个DADA视频 (10秒段模式)")
    
    processed_count = 0
    total_ghost_events = 0
    
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"👻 WiseAD处理视频 {i}/{len(video_files)}: {os.path.basename(video_path)}")
        
        # 检测鬼探头
        ghost_results = detect_ghost_probing_10s_segments(model, video_path)
        
        if ghost_results:
            # 保存结果
            detail_file, simple_file = save_ghost_results(ghost_results)
            
            if detail_file and simple_file:
                processed_count += 1
                total_ghost_events += ghost_results["ghost_summary"]["total_ghost_events"]
        else:
            logger.error(f"❌ 视频处理失败: {os.path.basename(video_path)}")
    
    # 6. 生成总结报告
    logger.info(f"🎉 WiseAD鬼探头检测完成!")
    logger.info(f"📊 处理统计:")
    logger.info(f"   - 成功处理视频: {processed_count}/{len(video_files)}")
    logger.info(f"   - 总鬼探头事件: {total_ghost_events}")
    logger.info(f"   - 平均每视频事件: {total_ghost_events/processed_count:.1f}" if processed_count > 0 else "   - 平均每视频事件: 0")
    
    # 确保日志文件保存到outputs
    outputs_dir = "/tmp/outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 复制详细日志到outputs
    import shutil
    if os.path.exists("/tmp/wisead_ghost_detailed.log"):
        shutil.copy("/tmp/wisead_ghost_detailed.log", "/tmp/outputs/wisead_detailed.log")
        logger.info(f"📋 详细日志已保存到: /tmp/outputs/wisead_detailed.log")
    
    # 复制结果文件到outputs
    if os.path.exists("/tmp/wisead_results"):
        shutil.copytree("/tmp/wisead_results", "/tmp/outputs/wisead_results", dirs_exist_ok=True)
        logger.info(f"📁 结果文件已保存到: /tmp/outputs/wisead_results/")

if __name__ == "__main__":
    main() 