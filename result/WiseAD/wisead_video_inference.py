#!/usr/bin/env python3
"""
WiseAD 视频推理系统
基于YOLO的自动驾驶场景视频分析
支持目标检测、行为分析和安全评估
优化A100 GPU性能 - 支持Azure Storage视频下载
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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """安装必要依赖 - 改进版本"""
    try:
        logger.info("🔧 开始安装必要依赖...")
        
        # 核心依赖包列表（精确版本）
        packages = [
            "ultralytics>=8.0.0",
            "opencv-python-headless>=4.5.0",
            "torch>=1.13.0",
            "torchvision>=0.14.0",
            "numpy>=1.21.0",
            "Pillow>=8.0.0",
            "azure-storage-blob>=12.0.0"  # 添加Azure Storage支持
        ]
        
        # 逐个安装依赖，确保成功
        for package in packages:
            logger.info(f"📦 安装 {package}...")
            try:
                # 使用更稳定的安装方式
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "--upgrade", "--no-cache-dir", package
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info(f"✅ {package} 安装成功")
                else:
                    logger.warning(f"⚠️ {package} 安装可能有问题: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"❌ {package} 安装超时")
            except Exception as e:
                logger.error(f"❌ {package} 安装异常: {e}")
        
        # 验证关键模块
        logger.info("🔍 验证关键模块...")
        test_imports = [
            ("cv2", "OpenCV"),
            ("torch", "PyTorch"), 
            ("ultralytics", "YOLO"),
            ("numpy", "NumPy"),
            ("azure.storage.blob", "Azure Storage")
        ]
        
        for module, name in test_imports:
            try:
                __import__(module)
                logger.info(f"✅ {name} 验证成功")
            except ImportError as e:
                logger.error(f"❌ {name} 验证失败: {e}")
                return False
        
        logger.info("✅ 所有依赖安装和验证完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 依赖安装失败: {e}")
        return False

# 导入其他模块（在依赖安装后）
def import_modules():
    """安全导入模块"""
    try:
        global cv2, torch
        import cv2
        import torch
        logger.info("✅ 模块导入成功")
        return True
    except ImportError as e:
        logger.error(f"❌ 模块导入失败: {e}")
        return False

class WiseADVideoInference:
    """WiseAD视频推理主类"""
    
    def __init__(self, config_path="wisead_config.json"):
        """初始化推理系统"""
        self.config = self.load_config(config_path)
        self.model = None
        self.device = None
        self.azure_client = None
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee'
        ]
        
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
        
        # 默认配置
        default_config = {
            "batch_size": 4,
            "confidence_threshold": 0.5,
            "model_type": "yolov8",
            "max_videos": 10,
            "azure_storage_container": "wisead-videos"
        }
        logger.info("📝 使用默认配置")
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
                logger.warning("⚠️ 未找到Azure Storage连接字符串，将只使用本地视频")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Azure Storage客户端初始化失败: {e}")
            return False
    
    def download_videos_from_azure(self, max_videos=10):
        """从Azure Storage下载视频到临时目录"""
        if not self.azure_client:
            return []
        
        try:
            container_name = self.config.get("azure_storage_container", "wisead-videos")
            logger.info(f"📥 从Azure Storage容器下载视频: {container_name}")
            
            container_client = self.azure_client.get_container_client(container_name)
            
            # 获取视频blob列表
            video_blobs = []
            for blob in container_client.list_blobs():
                if blob.name.endswith('.avi') and any(
                    blob.name.startswith(f'images_{i}_') for i in range(1, 6)
                ):
                    video_blobs.append(blob)
            
            # 按名称排序并限制数量
            video_blobs.sort(key=lambda x: x.name)
            video_blobs = video_blobs[:max_videos]
            
            logger.info(f"📹 找到 {len(video_blobs)} 个视频文件需要下载")
            
            # 创建临时下载目录
            download_dir = tempfile.mkdtemp(prefix="wisead_videos_")
            downloaded_videos = []
            
            for i, blob in enumerate(video_blobs, 1):
                try:
                    logger.info(f"📥 下载视频 {i}/{len(video_blobs)}: {blob.name}")
                    
                    # 下载到临时文件
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
            
            logger.info(f"📥 Azure Storage视频下载完成: {len(downloaded_videos)}/{len(video_blobs)}")
            return downloaded_videos
            
        except Exception as e:
            logger.error(f"❌ Azure Storage视频下载失败: {e}")
            return []
    
    def setup_device(self):
        """设置计算设备"""
        try:
            if torch.cuda.is_available():
                self.device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🚀 使用GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                
                # A100优化设置
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()
                
            else:
                self.device = 'cpu'
                logger.warning("⚠️ GPU不可用，使用CPU")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 设备设置失败: {e}")
            self.device = 'cpu'
            return False
    
    def load_model(self):
        """加载YOLO模型"""
        try:
            logger.info("🤖 加载YOLO模型...")
            
            # 导入ultralytics
            from ultralytics import YOLO
            
            # 根据配置选择模型
            model_type = self.config.get("model_type", "yolov8")
            if model_type == "yolov8":
                # 使用更大的模型以充分利用A100性能
                model_name = "yolov8s.pt"  # small版本，平衡速度和精度
            else:
                model_name = "yolov8s.pt"
            
            # 加载模型
            self.model = YOLO(model_name)
            logger.info(f"✅ {model_name} 模型加载成功")
            
            # 设置设备
            if self.device:
                self.model.to(self.device)
                logger.info(f"🎯 模型已转移到: {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def find_local_videos(self, max_videos=5):
        """查找本地视频文件"""
        try:
            logger.info("🔍 搜索本地视频文件...")
            
            video_files = []
            search_dirs = ["test_video", "DADA-2000-videos", "frames", "."]
            video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.wmv']
            
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    logger.info(f"📁 搜索目录: {search_dir}")
                    
                    for ext in video_extensions:
                        pattern = f"**/*{ext}"
                        found_files = list(Path(search_dir).glob(pattern))
                        video_files.extend([str(f) for f in found_files])
                    
                    if len(video_files) >= max_videos:
                        break
            
            # 限制视频数量
            video_files = video_files[:max_videos]
            
            logger.info(f"📹 找到 {len(video_files)} 个视频文件:")
            for i, video in enumerate(video_files, 1):
                try:
                    file_size = os.path.getsize(video) / 1024 / 1024  # MB
                    logger.info(f"   {i}. {Path(video).name} ({file_size:.1f}MB)")
                except:
                    logger.info(f"   {i}. {Path(video).name}")
            
            return video_files
            
        except Exception as e:
            logger.error(f"❌ 视频搜索失败: {e}")
            return []
    
    def get_videos_to_process(self):
        """获取要处理的视频文件（Azure + 本地）"""
        max_videos = self.config.get("max_videos", 10)
        
        # 1. 尝试从Azure Storage下载视频
        azure_videos = []
        if self.azure_client:
            azure_videos = self.download_videos_from_azure(max_videos)
        
        # 2. 如果Azure视频不足，补充本地视频
        remaining_slots = max_videos - len(azure_videos)
        local_videos = []
        if remaining_slots > 0:
            local_videos = self.find_local_videos(remaining_slots)
        
        # 3. 合并视频列表
        all_videos = azure_videos + local_videos
        
        logger.info(f"🎬 视频来源统计:")
        logger.info(f"   Azure Storage: {len(azure_videos)} 个")
        logger.info(f"   本地文件: {len(local_videos)} 个")
        logger.info(f"   总计: {len(all_videos)} 个")
        
        return all_videos
    
    def process_video(self, video_path):
        """处理单个视频文件"""
        if not self.model:
            logger.error("❌ 模型未加载")
            return None
        
        try:
            video_name = Path(video_path).name
            logger.info(f"🎬 开始处理视频: {video_name}")
            
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"❌ 无法打开视频: {video_path}")
                return None
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"📊 视频信息: {total_frames}帧, {fps}FPS, {width}x{height}, 时长{duration:.1f}秒")
            
            # 分析结果存储
            analysis_results = {
                "video_info": {
                    "path": video_path,
                    "name": video_name,
                    "total_frames": total_frames,
                    "fps": fps,
                    "dimensions": [width, height],
                    "duration_seconds": duration
                },
                "detection_summary": {
                    "total_detections": 0,
                    "vehicle_count": 0,
                    "pedestrian_count": 0,
                    "traffic_elements": 0
                },
                "frame_detections": [],
                "processing_stats": {
                    "frames_analyzed": 0,
                    "processing_time": 0,
                    "fps_processed": 0
                }
            }
            
            frame_count = 0
            frames_analyzed = 0
            start_time = datetime.now()
            
            # 分析间隔：A100可以处理更频繁的帧
            analysis_interval = max(1, fps // 4)  # 每0.25秒分析一次
            
            # 批处理设置
            batch_size = self.config.get("batch_size", 4)
            frame_batch = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 跳过不需要分析的帧
                if frame_count % analysis_interval != 0:
                    continue
                
                frame_batch.append((frame_count, frame))
                
                # 当达到批处理大小或视频结束时进行推理
                if len(frame_batch) >= batch_size or frame_count >= total_frames:
                    self.process_frame_batch(frame_batch, analysis_results)
                    frames_analyzed += len(frame_batch)
                    frame_batch = []
                
                # 进度显示
                if frame_count % (fps * 5) == 0:  # 每5秒显示一次进度
                    progress = (frame_count / total_frames) * 100
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.info(f"📈 处理进度: {progress:.1f}% ({frame_count}/{total_frames}帧) - 已用时{elapsed:.1f}秒")
            
            cap.release()
            
            # 计算最终统计
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            analysis_results["processing_stats"] = {
                "frames_analyzed": frames_analyzed,
                "processing_time": processing_time,
                "fps_processed": frames_analyzed / processing_time if processing_time > 0 else 0
            }
            
            logger.info(f"✅ 视频分析完成: {video_name}")
            logger.info(f"📊 总检测数: {analysis_results['detection_summary']['total_detections']}")
            logger.info(f"🚗 车辆检测: {analysis_results['detection_summary']['vehicle_count']}")
            logger.info(f"🚶 行人检测: {analysis_results['detection_summary']['pedestrian_count']}")
            logger.info(f"⏱️ 处理时间: {processing_time:.1f}秒 ({frames_analyzed}帧)")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ 视频处理失败 {video_path}: {e}")
            return None
    
    def process_frame_batch(self, frame_batch, analysis_results):
        """批处理帧分析"""
        try:
            if not frame_batch:
                return
            
            frame_numbers = [num for num, _ in frame_batch]
            
            # 处理每帧（逐个处理以避免批处理问题）
            for frame_num, frame in frame_batch:
                try:
                    # 将BGR转为RGB（OpenCV使用BGR，PIL使用RGB）
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 单帧推理
                    results = self.model(frame_rgb, conf=self.config.get("confidence_threshold", 0.5))
                    
                    # 处理检测结果
                    frame_detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                # 提取检测信息
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                class_id = int(box.cls[0].cpu().numpy())
                                
                                if class_id < len(self.class_names):
                                    class_name = self.class_names[class_id]
                                    
                                    detection = {
                                        "frame": frame_num,
                                        "class": class_name,
                                        "confidence": float(confidence),
                                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                                    }
                                    frame_detections.append(detection)
                                    
                                    # 统计计数
                                    analysis_results["detection_summary"]["total_detections"] += 1
                                    
                                    if class_name in ['car', 'bus', 'truck', 'motorcycle']:
                                        analysis_results["detection_summary"]["vehicle_count"] += 1
                                    elif class_name == 'person':
                                        analysis_results["detection_summary"]["pedestrian_count"] += 1
                                    elif class_name in ['traffic light', 'stop sign']:
                                        analysis_results["detection_summary"]["traffic_elements"] += 1
                    
                    if frame_detections:
                        analysis_results["frame_detections"].extend(frame_detections)
                        
                except Exception as frame_error:
                    logger.warning(f"⚠️ 帧 {frame_num} 处理失败: {frame_error}")
                    continue
                
        except Exception as e:
            logger.error(f"❌ 批处理失败: {e}")
    
    def save_results(self, results, output_dir):
        """保存分析结果"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成结果文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = Path(results["video_info"]["name"]).stem
            result_file = os.path.join(output_dir, f"wisead_analysis_{video_name}_{timestamp}.json")
            
            # 保存JSON结果
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 结果已保存: {result_file}")
            return result_file
            
        except Exception as e:
            logger.error(f"❌ 结果保存失败: {e}")
            return None
    
    def run_inference(self):
        """运行完整的推理流程"""
        logger.info("🚀 启动WiseAD视频推理系统 (低优先级A100优化版)")
        
        # 1. 安装依赖
        if not install_dependencies():
            logger.error("❌ 依赖安装失败，退出")
            return False
        
        # 2. 导入模块
        if not import_modules():
            logger.error("❌ 模块导入失败，退出")
            return False
        
        # 3. 设置Azure客户端
        self.setup_azure_client()
        
        # 4. 设置设备
        self.setup_device()
        
        # 5. 加载模型
        if not self.load_model():
            logger.error("❌ 模型加载失败，退出")
            return False
        
        # 6. 创建工作目录
        work_dir = tempfile.mkdtemp(prefix="wisead_lowpri_")
        results_dir = os.path.join(work_dir, "results")
        
        try:
            # 7. 获取要处理的视频文件（Azure + 本地）
            video_files = self.get_videos_to_process()
            
            if not video_files:
                logger.error("❌ 未找到视频文件（Azure Storage 和本地都没有）")
                return False
            
            # 8. 处理每个视频
            all_results = []
            for i, video_file in enumerate(video_files, 1):
                logger.info(f"🎬 处理视频 {i}/{len(video_files)}: {Path(video_file).name}")
                
                result = self.process_video(video_file)
                if result:
                    # 保存结果
                    result_file = self.save_results(result, results_dir)
                    if result_file:
                        all_results.append(result)
                        logger.info(f"✅ 视频 {i} 处理完成")
                else:
                    logger.warning(f"⚠️ 视频 {i} 处理失败")
            
            # 9. 生成总结报告
            if all_results:
                summary = self.generate_summary_report(all_results)
                summary_file = os.path.join(results_dir, f"wisead_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                
                logger.info("🎉 WiseAD视频推理完成！")
                logger.info(f"📊 成功处理视频: {len(all_results)}/{len(video_files)}")
                logger.info(f"📁 结果目录: {results_dir}")
                
                # 显示处理统计
                total_detections = sum(r["detection_summary"]["total_detections"] for r in all_results)
                total_vehicles = sum(r["detection_summary"]["vehicle_count"] for r in all_results)
                total_pedestrians = sum(r["detection_summary"]["pedestrian_count"] for r in all_results)
                
                logger.info(f"🎯 总体统计:")
                logger.info(f"   - 总检测数: {total_detections}")
                logger.info(f"   - 车辆数量: {total_vehicles}")
                logger.info(f"   - 行人数量: {total_pedestrians}")
                
                return True
            else:
                logger.error("❌ 所有视频处理失败")
                return False
            
        finally:
            # 清理工作目录
            if os.path.exists(work_dir):
                try:
                    # 复制重要结果到当前目录
                    if os.path.exists(results_dir):
                        final_results_dir = "wisead_results"
                        if os.path.exists(final_results_dir):
                            shutil.rmtree(final_results_dir)
                        shutil.copytree(results_dir, final_results_dir)
                        logger.info(f"📋 结果已复制到: {final_results_dir}")
                    
                    shutil.rmtree(work_dir)
                    logger.info(f"🧹 清理工作目录: {work_dir}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理失败: {e}")
    
    def generate_summary_report(self, all_results):
        """生成总结报告"""
        summary = {
            "report_info": {
                "timestamp": datetime.now().isoformat(),
                "system": "WiseAD Video Inference System",
                "model": "YOLOv8 on Low Priority A100 GPU",
                "version": "2.2 (Azure Storage支持)"
            },
            "processing_summary": {
                "total_videos": len(all_results),
                "total_detections": sum(r["detection_summary"]["total_detections"] for r in all_results),
                "total_vehicles": sum(r["detection_summary"]["vehicle_count"] for r in all_results),
                "total_pedestrians": sum(r["detection_summary"]["pedestrian_count"] for r in all_results),
                "total_traffic_elements": sum(r["detection_summary"]["traffic_elements"] for r in all_results)
            },
            "performance_stats": {
                "total_processing_time": sum(r["processing_stats"]["processing_time"] for r in all_results),
                "average_fps": np.mean([r["processing_stats"]["fps_processed"] for r in all_results if r["processing_stats"]["fps_processed"] > 0]),
                "total_frames_analyzed": sum(r["processing_stats"]["frames_analyzed"] for r in all_results)
            },
            "video_details": []
        }
        
        for result in all_results:
            video_summary = {
                "video_name": result["video_info"]["name"],
                "duration": result["video_info"]["duration_seconds"],
                "total_detections": result["detection_summary"]["total_detections"],
                "vehicles": result["detection_summary"]["vehicle_count"],
                "pedestrians": result["detection_summary"]["pedestrian_count"],
                "processing_time": result["processing_stats"]["processing_time"],
                "processing_fps": result["processing_stats"]["fps_processed"]
            }
            summary["video_details"].append(video_summary)
        
        return summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WiseAD视频推理系统 - 低优先级A100优化版 + Azure Storage支持")
    parser.add_argument("--config", type=str, default="wisead_config.json", help="配置文件路径")
    
    args = parser.parse_args()
    
    # 创建推理系统
    wisead = WiseADVideoInference(args.config)
    
    # 运行推理
    success = wisead.run_inference()
    
    if success:
        logger.info("✅ WiseAD推理任务成功完成")
        sys.exit(0)
    else:
        logger.error("❌ WiseAD推理任务失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 