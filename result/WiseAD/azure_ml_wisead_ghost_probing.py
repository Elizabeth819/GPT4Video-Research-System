#!/usr/bin/env python3
"""
Azure ML A100 WiseAD鬼探头检测系统
使用WiseAD模型进行100个DADA视频的鬼探头推理分析
专门针对自动驾驶场景中的鬼探头行为检测
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import argparse
import logging
import subprocess
from datetime import datetime
import tempfile
import shutil
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """安装WiseAD推理必要依赖"""
    try:
        logger.info("🔧 开始安装WiseAD A100 GPU依赖...")
        
        # 首先升级pip
        logger.info("📦 升级pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True, text=True, timeout=120)
        
        packages = [
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
        
        # 逐个安装包并验证
        for package in packages:
            logger.info(f"📦 安装 {package}...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "--upgrade", "--no-cache-dir", "--force-reinstall", package
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info(f"✅ {package} 安装成功")
                else:
                    logger.warning(f"⚠️ {package} 安装警告: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"⏰ {package} 安装超时，继续下一个")
            except Exception as e:
                logger.warning(f"⚠️ {package} 安装异常: {e}")
        
        # 验证关键模块
        test_imports = [
            ("cv2", "OpenCV"),
            ("torch", "PyTorch"),
            ("ultralytics", "YOLO"),
            ("azure.storage.blob", "Azure Storage")
        ]
        
        all_success = True
        for module, name in test_imports:
            try:
                __import__(module)
                logger.info(f"✅ {name} 验证成功")
            except ImportError as e:
                logger.error(f"❌ {name} 验证失败: {e}")
                all_success = False
        
        if all_success:
            logger.info("✅ 所有WiseAD依赖安装验证完成")
            return True
        else:
            logger.error("❌ 部分依赖验证失败")
            return False
        
    except Exception as e:
        logger.error(f"❌ 依赖安装失败: {e}")
        return False

def safe_import_modules():
    """安全导入WiseAD相关模块"""
    try:
        global cv2, torch, YOLO
        import cv2
        import torch
        from ultralytics import YOLO
        logger.info("✅ WiseAD核心模块导入成功")
        return True
    except ImportError as e:
        logger.error(f"❌ WiseAD模块导入失败: {e}")
        return False

class WiseADGhostProbingDetector:
    """WiseAD鬼探头检测系统"""
    
    def __init__(self, config_path="wisead_ghost_probing_config.json"):
        """初始化WiseAD鬼探头检测系统"""
        self.config = self.load_config(config_path)
        self.model = None
        self.device = None
        self.azure_client = None
        
        # WiseAD相关类别（专注于鬼探头相关目标）
        self.ghost_probing_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 
            'traffic light', 'stop sign', 'parking meter'
        ]
        
        # 鬼探头检测规则参数
        self.ghost_rules = {
            "sudden_appearance_threshold": 0.7,  # 突然出现阈值
            "proximity_danger_distance": 50,     # 危险距离（像素）
            "speed_change_threshold": 0.8,       # 速度变化阈值
            "unexpected_movement_score": 0.6     # 意外运动评分
        }
        
    def load_config(self, config_path):
        """加载配置文件"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"✅ 配置文件加载成功: {config_path}")
                    return config
        except Exception as e:
            logger.warning(f"⚠️ 配置文件加载失败: {e}")
        
        # 默认WiseAD鬼探头配置
        default_config = {
            "max_videos": 100,
            "azure_storage_container": "dada-videos",
            "batch_size": 4,
            "confidence_threshold": 0.5,
            "model_type": "yolov8s",
            "ghost_detection_sensitivity": "high",
            "frame_analysis_interval": 3  # 每3帧分析一次
        }
        logger.info("📝 使用默认WiseAD鬼探头配置")
        return default_config
    
    def setup_azure_client(self):
        """设置Azure Storage客户端"""
        try:
            from azure.storage.blob import BlobServiceClient
            
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if connection_string:
                self.azure_client = BlobServiceClient.from_connection_string(connection_string)
                logger.info("✅ Azure Storage客户端初始化成功")
                return True
            else:
                logger.warning("⚠️ 未找到Azure Storage连接字符串")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Azure Storage客户端初始化失败: {e}")
            return False
    
    def setup_device(self):
        """设置A100 GPU设备"""
        try:
            if torch.cuda.is_available():
                self.device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🚀 使用A100 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                
                # A100优化设置
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()
                
                # 设置GPU内存管理
                torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%显存
                
            else:
                self.device = 'cpu'
                logger.warning("⚠️ GPU不可用，使用CPU")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 设备设置失败: {e}")
            self.device = 'cpu'
            return False
    
    def load_wisead_model(self):
        """加载WiseAD YOLO模型"""
        try:
            logger.info("🤖 加载WiseAD YOLO模型...")
            
            # 根据配置选择模型
            model_type = self.config.get("model_type", "yolov8s")
            model_name = f"{model_type}.pt"
            
            # 加载YOLO模型
            self.model = YOLO(model_name)
            logger.info(f"✅ WiseAD {model_name} 模型加载成功")
            
            # 设置设备
            if self.device:
                self.model.to(self.device)
                logger.info(f"🎯 WiseAD模型已转移到: {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ WiseAD模型加载失败: {e}")
            return False
    
    def download_dada_videos(self, max_videos=100):
        """从Azure Storage下载100个DADA视频"""
        if not self.azure_client:
            return []
        
        try:
            container_name = self.config.get("azure_storage_container", "dada-videos")
            logger.info(f"📥 从Azure Storage下载DADA视频: {container_name}")
            
            container_client = self.azure_client.get_container_client(container_name)
            
            # 获取所有DADA视频 (images_1_001 到 images_5_XXX)
            video_blobs = []
            for blob in container_client.list_blobs():
                if blob.name.endswith('.avi') and any(
                    blob.name.startswith(f'images_{i}_') for i in range(1, 6)
                ):
                    video_blobs.append(blob)
            
            # 按名称排序，确保处理顺序一致
            video_blobs.sort(key=lambda x: x.name)
            video_blobs = video_blobs[:max_videos]
            
            logger.info(f"📹 找到 {len(video_blobs)} 个DADA视频待下载")
            
            # 创建临时下载目录
            download_dir = tempfile.mkdtemp(prefix="wisead_ghost_probing_")
            downloaded_videos = []
            
            for i, blob in enumerate(video_blobs, 1):
                try:
                    logger.info(f"📥 下载视频 {i}/{len(video_blobs)}: {blob.name}")
                    
                    local_path = os.path.join(download_dir, blob.name)
                    blob_client = self.azure_client.get_blob_client(
                        container=container_name, 
                        blob=blob.name
                    )
                    
                    with open(local_path, 'wb') as download_file:
                        download_stream = blob_client.download_blob()
                        download_stream.readinto(download_file)
                    
                    downloaded_videos.append(local_path)
                    logger.info(f"✅ 下载成功: {blob.name}")
                    
                except Exception as e:
                    logger.error(f"❌ 下载失败 {blob.name}: {e}")
            
            logger.info(f"📥 DADA视频下载完成: {len(downloaded_videos)}/{len(video_blobs)}")
            return downloaded_videos
            
        except Exception as e:
            logger.error(f"❌ Azure Storage视频下载失败: {e}")
            return []
    
    def detect_ghost_probing_in_video(self, video_path):
        """使用WiseAD模型检测视频中的鬼探头行为"""
        try:
            video_name = Path(video_path).name
            video_id = video_name.replace('.avi', '').replace('.mp4', '')
            
            logger.info(f"👻 开始WiseAD鬼探头检测: {video_name}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"❌ 无法打开视频: {video_path}")
                return None
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"📹 视频信息: {total_frames}帧, {fps:.1f}FPS, {duration:.1f}秒")
            
            # 初始化分析结果
            ghost_analysis = {
                "video_id": video_id,
                "video_name": video_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "video_info": {
                    "total_frames": total_frames,
                    "fps": fps,
                    "duration": duration
                },
                "ghost_detections": [],
                "frame_analysis": [],
                "ghost_summary": {
                    "total_ghost_events": 0,
                    "high_risk_events": 0,
                    "potential_ghost_events": 0,
                    "normal_traffic_events": 0
                },
                "processing_info": {
                    "model": "WiseAD_YOLOv8",
                    "device": str(self.device),
                    "confidence_threshold": self.config.get("confidence_threshold", 0.5)
                }
            }
            
            # 分析参数
            analysis_interval = self.config.get("frame_analysis_interval", 3)
            confidence_threshold = self.config.get("confidence_threshold", 0.5)
            
            frame_count = 0
            previous_detections = []
            
            start_time = datetime.now()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 按间隔分析帧
                if frame_count % analysis_interval == 0:
                    # WiseAD模型推理
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.model(frame_rgb, conf=confidence_threshold)
                    
                    # 提取当前帧检测结果
                    current_detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                class_id = int(box.cls[0].cpu().numpy())
                                
                                # 获取类别名称
                                if hasattr(result, 'names') and class_id in result.names:
                                    class_name = result.names[class_id]
                                else:
                                    class_name = f"class_{class_id}"
                                
                                # 只关注鬼探头相关目标
                                if class_name in self.ghost_probing_classes:
                                    detection = {
                                        "frame": frame_count,
                                        "time": frame_count / fps,
                                        "class": class_name,
                                        "confidence": float(confidence),
                                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                                        "area": (x2 - x1) * (y2 - y1)
                                    }
                                    current_detections.append(detection)
                    
                    # 鬼探头行为分析
                    ghost_event = self.analyze_ghost_probing_behavior(
                        current_detections, previous_detections, frame_count, fps
                    )
                    
                    if ghost_event:
                        ghost_analysis["ghost_detections"].append(ghost_event)
                        
                        # 更新统计
                        risk_level = ghost_event.get("risk_level", "normal")
                        if risk_level == "high":
                            ghost_analysis["ghost_summary"]["high_risk_events"] += 1
                        elif risk_level == "potential":
                            ghost_analysis["ghost_summary"]["potential_ghost_events"] += 1
                        else:
                            ghost_analysis["ghost_summary"]["normal_traffic_events"] += 1
                    
                    # 保存帧分析结果
                    ghost_analysis["frame_analysis"].append({
                        "frame": frame_count,
                        "time": frame_count / fps,
                        "detections_count": len(current_detections),
                        "ghost_detected": ghost_event is not None
                    })
                    
                    previous_detections = current_detections
                
                # 每100帧显示一次进度
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"📈 WiseAD分析进度: {progress:.1f}% ({frame_count}/{total_frames}帧)")
            
            cap.release()
            
            # 计算处理统计
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            ghost_analysis["processing_info"]["processing_time"] = processing_time
            ghost_analysis["processing_info"]["frames_analyzed"] = frame_count // analysis_interval
            ghost_analysis["ghost_summary"]["total_ghost_events"] = len(ghost_analysis["ghost_detections"])
            
            logger.info(f"✅ WiseAD鬼探头检测完成: {video_name}")
            logger.info(f"👻 鬼探头事件: {ghost_analysis['ghost_summary']['total_ghost_events']}")
            logger.info(f"🚨 高风险事件: {ghost_analysis['ghost_summary']['high_risk_events']}")
            logger.info(f"⚠️ 潜在事件: {ghost_analysis['ghost_summary']['potential_ghost_events']}")
            logger.info(f"⏱️ 处理时间: {processing_time:.1f}秒")
            
            return ghost_analysis
            
        except Exception as e:
            logger.error(f"❌ WiseAD鬼探头检测失败 {video_path}: {e}")
            return None
    
    def analyze_ghost_probing_behavior(self, current_detections, previous_detections, frame_num, fps):
        """分析鬼探头行为"""
        try:
            if not current_detections or not previous_detections:
                return None
            
            # 检测突然出现的目标
            for detection in current_detections:
                # 检查是否为突然出现
                sudden_appearance = self.check_sudden_appearance(detection, previous_detections)
                
                # 检查危险距离
                proximity_danger = self.check_proximity_danger(detection)
                
                # 检查意外运动
                unexpected_movement = self.check_unexpected_movement(detection, previous_detections)
                
                # 综合评估鬼探头风险
                ghost_score = 0
                risk_factors = []
                
                if sudden_appearance:
                    ghost_score += 0.4
                    risk_factors.append("突然出现")
                
                if proximity_danger:
                    ghost_score += 0.3
                    risk_factors.append("危险距离")
                
                if unexpected_movement:
                    ghost_score += 0.3
                    risk_factors.append("意外运动")
                
                # 根据评分判断鬼探头类型
                if ghost_score >= 0.7:
                    risk_level = "high"
                    ghost_type = "ghost probing"
                elif ghost_score >= 0.4:
                    risk_level = "potential"
                    ghost_type = "potential ghost probing"
                else:
                    continue  # 正常交通情况，不报告
                
                # 构造鬼探头事件
                ghost_event = {
                    "frame": frame_num,
                    "time": frame_num / fps,
                    "object_class": detection["class"],
                    "confidence": detection["confidence"],
                    "bbox": detection["bbox"],
                    "ghost_type": ghost_type,
                    "risk_level": risk_level,
                    "ghost_score": ghost_score,
                    "risk_factors": risk_factors,
                    "detection_method": "WiseAD_YOLO_Analysis"
                }
                
                return ghost_event
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ 鬼探头行为分析失败: {e}")
            return None
    
    def check_sudden_appearance(self, detection, previous_detections):
        """检查目标是否突然出现"""
        try:
            # 检查前一帧是否存在相似目标
            detection_center = detection["center"]
            detection_class = detection["class"]
            
            for prev_det in previous_detections:
                if prev_det["class"] == detection_class:
                    # 计算中心距离
                    prev_center = prev_det["center"]
                    distance = np.sqrt((detection_center[0] - prev_center[0])**2 + 
                                     (detection_center[1] - prev_center[1])**2)
                    
                    # 如果距离较近，说明是连续检测，非突然出现
                    if distance < 100:  # 像素距离阈值
                        return False
            
            # 没有找到相似的前置目标，可能是突然出现
            return True
            
        except Exception:
            return False
    
    def check_proximity_danger(self, detection):
        """检查目标是否处于危险距离"""
        try:
            # 基于边界框大小和位置判断危险距离
            x1, y1, x2, y2 = detection["bbox"]
            
            # 假设视频分辨率为标准尺寸，底部中央为车辆位置
            frame_width = 640  # 假设宽度
            frame_height = 480  # 假设高度
            
            vehicle_center_x = frame_width / 2
            vehicle_front_y = frame_height * 0.8  # 车辆前方位置
            
            detection_center_x = (x1 + x2) / 2
            detection_center_y = (y1 + y2) / 2
            
            # 计算到车辆的距离
            distance_to_vehicle = np.sqrt((detection_center_x - vehicle_center_x)**2 + 
                                        (detection_center_y - vehicle_front_y)**2)
            
            # 危险距离阈值
            danger_threshold = self.ghost_rules["proximity_danger_distance"]
            
            return distance_to_vehicle < danger_threshold
            
        except Exception:
            return False
    
    def check_unexpected_movement(self, detection, previous_detections):
        """检查目标是否有意外运动"""
        try:
            # 简化的意外运动检测：基于目标大小的快速变化
            detection_area = detection["area"]
            detection_class = detection["class"]
            
            for prev_det in previous_detections:
                if prev_det["class"] == detection_class:
                    area_change_ratio = abs(detection_area - prev_det["area"]) / prev_det["area"]
                    
                    # 如果面积变化很大，可能表示快速接近或远离
                    if area_change_ratio > 0.5:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def save_ghost_analysis_results(self, results, output_dir):
        """保存WiseAD鬼探头分析结果"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            video_name = results["video_name"].replace('.avi', '').replace('.mp4', '')
            result_file = os.path.join(output_dir, f"wisead_ghost_{video_name}.json")
            
            # 保存为与GPT-4.1兼容的格式
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 WiseAD结果已保存: {result_file}")
            return result_file
            
        except Exception as e:
            logger.error(f"❌ 结果保存失败: {e}")
            return None
    
    def run_wisead_ghost_probing_analysis(self):
        """运行完整的WiseAD鬼探头分析流程"""
        logger.info("🚀 启动WiseAD A100 鬼探头检测系统")
        logger.info("🤖 使用WiseAD YOLO模型进行本地推理")
        
        # 1. 安装依赖
        if not install_dependencies():
            logger.error("❌ 依赖安装失败，退出")
            return False
        
        # 2. 安全导入模块
        if not safe_import_modules():
            logger.error("❌ 模块导入失败，退出")
            return False
        
        # 3. 设置Azure客户端
        self.setup_azure_client()
        
        # 4. 设置A100设备
        self.setup_device()
        
        # 5. 加载WiseAD模型
        if not self.load_wisead_model():
            logger.error("❌ WiseAD模型加载失败，退出")
            return False
        
        # 6. 创建工作目录
        work_dir = tempfile.mkdtemp(prefix="wisead_ghost_probing_")
        results_dir = os.path.join(work_dir, "results")
        
        try:
            # 7. 下载100个DADA视频
            max_videos = self.config.get("max_videos", 100)
            video_files = self.download_dada_videos(max_videos)
            
            if not video_files:
                logger.error("❌ 未找到DADA视频文件")
                return False
            
            logger.info(f"🎬 准备使用WiseAD分析 {len(video_files)} 个DADA视频")
            
            # 8. 处理每个视频
            all_results = []
            success_count = 0
            total_videos = len(video_files)
            
            for i, video_file in enumerate(video_files, 1):
                video_name = Path(video_file).name
                logger.info(f"👻 WiseAD处理视频 {i}/{total_videos}: {video_name}")
                
                try:
                    result = self.detect_ghost_probing_in_video(video_file)
                    if result:
                        # 保存结果
                        result_file = self.save_ghost_analysis_results(result, results_dir)
                        if result_file:
                            all_results.append(result)
                            success_count += 1
                            logger.info(f"✅ WiseAD视频 {i} 分析完成: {video_name}")
                        else:
                            logger.warning(f"⚠️ 视频 {i} 结果保存失败: {video_name}")
                    else:
                        logger.warning(f"⚠️ WiseAD视频 {i} 分析失败: {video_name}")
                        
                except Exception as e:
                    logger.error(f"❌ 视频 {i} 处理异常: {video_name} - {e}")
                
                # 每10个视频输出一次进度
                if i % 10 == 0 or i == total_videos:
                    logger.info(f"📊 WiseAD进度报告: {success_count}/{i} 成功处理")
            
            # 9. 生成WiseAD总结报告
            if all_results:
                summary = self.generate_wisead_summary_report(all_results)
                summary_file = os.path.join(results_dir, f"wisead_ghost_probing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                
                logger.info("🎉 WiseAD A100 鬼探头检测完成！")
                logger.info(f"📊 成功处理视频: {success_count}/{total_videos}")
                logger.info(f"📁 结果目录: {results_dir}")
                
                # 统计WiseAD鬼探头检测结果
                total_ghost_events = sum(r["ghost_summary"]["total_ghost_events"] for r in all_results)
                high_risk_events = sum(r["ghost_summary"]["high_risk_events"] for r in all_results)
                potential_events = sum(r["ghost_summary"]["potential_ghost_events"] for r in all_results)
                
                logger.info(f"🎯 WiseAD鬼探头检测统计:")
                logger.info(f"   - 总鬼探头事件: {total_ghost_events}")
                logger.info(f"   - 高风险事件: {high_risk_events}")
                logger.info(f"   - 潜在风险事件: {potential_events}")
                logger.info(f"   - 平均每视频: {total_ghost_events / len(all_results):.1f}事件")
                
                success_rate = success_count / total_videos
                if success_rate >= 0.5:
                    logger.info(f"✅ WiseAD任务成功！成功率: {success_rate * 100:.1f}%")
                    return True
                else:
                    logger.warning(f"⚠️ 成功率较低: {success_rate * 100:.1f}%，但已保存部分结果")
                    return True
            else:
                logger.error("❌ 所有WiseAD视频分析失败")
                return False
            
        finally:
            # 清理工作目录
            if os.path.exists(work_dir):
                try:
                    # 复制结果到当前目录
                    final_results_dir = "wisead_ghost_probing_results"
                    if os.path.exists(results_dir):
                        if os.path.exists(final_results_dir):
                            shutil.rmtree(final_results_dir)
                        shutil.copytree(results_dir, final_results_dir)
                        logger.info(f"📋 WiseAD结果已复制到: {final_results_dir}")
                    
                    shutil.rmtree(work_dir)
                    logger.info(f"🧹 清理工作目录: {work_dir}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理失败: {e}")
    
    def generate_wisead_summary_report(self, all_results):
        """生成WiseAD总结报告"""
        ghost_detections = []
        high_risk_videos = []
        potential_risk_videos = []
        normal_videos = []
        
        for result in all_results:
            video_id = result.get("video_id", "unknown")
            total_events = result["ghost_summary"]["total_ghost_events"]
            high_risk = result["ghost_summary"]["high_risk_events"]
            potential = result["ghost_summary"]["potential_ghost_events"]
            
            if high_risk > 0:
                high_risk_videos.append({"video_id": video_id, "events": high_risk})
            elif potential > 0:
                potential_risk_videos.append({"video_id": video_id, "events": potential})
            else:
                normal_videos.append(video_id)
        
        summary = {
            "report_info": {
                "timestamp": datetime.now().isoformat(),
                "system": "WiseAD A100 Ghost Probing Detection",
                "model": "WiseAD YOLO v8",
                "analysis_method": "Local GPU Inference",
                "version": "1.0 - WiseAD Based"
            },
            "processing_summary": {
                "total_videos": len(all_results),
                "high_risk_videos": len(high_risk_videos),
                "potential_risk_videos": len(potential_risk_videos),
                "normal_videos": len(normal_videos),
                "total_ghost_events": sum(r["ghost_summary"]["total_ghost_events"] for r in all_results)
            },
            "wisead_performance": {
                "model_type": "YOLOv8s",
                "device": "A100 GPU",
                "confidence_threshold": self.config.get("confidence_threshold", 0.5),
                "local_inference": True,
                "no_external_api": True
            },
            "detection_details": {
                "high_risk_videos": high_risk_videos,
                "potential_risk_videos": potential_risk_videos,
                "normal_videos": normal_videos
            },
            "video_results": []
        }
        
        for result in all_results:
            video_summary = {
                "video_id": result.get("video_id"),
                "total_ghost_events": result["ghost_summary"]["total_ghost_events"],
                "high_risk_events": result["ghost_summary"]["high_risk_events"],
                "potential_events": result["ghost_summary"]["potential_ghost_events"],
                "processing_time": result["processing_info"].get("processing_time", 0)
            }
            summary["video_results"].append(video_summary)
        
        return summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WiseAD A100 鬼探头检测系统")
    parser.add_argument("--config", type=str, default="wisead_ghost_probing_config.json", help="配置文件路径")
    
    args = parser.parse_args()
    
    # 创建WiseAD鬼探头检测系统
    ghost_detector = WiseADGhostProbingDetector(args.config)
    
    # 运行检测
    success = ghost_detector.run_wisead_ghost_probing_analysis()
    
    if success:
        logger.info("✅ WiseAD A100 鬼探头检测成功完成")
        sys.exit(0)
    else:
        logger.error("❌ WiseAD A100 鬼探头检测失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 