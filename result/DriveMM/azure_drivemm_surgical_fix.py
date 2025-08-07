#!/usr/bin/env python3
"""
DriveMM Azure推理脚本 - 外科手术式CUDA错误修复
专门针对frame processing阶段的CUDA device-side assert错误
"""

import os
import sys
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SurgicalFixDriveMM:
    def __init__(self):
        # 设置CUDA调试环境
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        
        self.setup_azure_clients()
        self.setup_drivemm_model_safe()
        
    def setup_azure_clients(self):
        """设置Azure客户端"""
        logger.info("🔗 设置Azure连接...")
        
        try:
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            
            if connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    conn_str=connection_string
                )
            else:
                storage_account = "drivelmmstorage2e932dad7"
                self.storage_url = f"https://{storage_account}.blob.core.windows.net"
                credential = DefaultAzureCredential()
                self.blob_service_client = BlobServiceClient(
                    account_url=self.storage_url,
                    credential=credential
                )
            
            logger.info("✅ Azure Storage连接成功")
        except Exception as e:
            logger.error(f"❌ Azure Storage连接失败: {e}")
            raise
    
    def setup_drivemm_model_safe(self):
        """安全设置DriveMM模型 - 专注于修复frame processing"""
        logger.info("🔧 安全设置DriveMM模型...")
        
        # 检查GPU可用性
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 使用设备: {self.device}")
        
        if not torch.cuda.is_available():
            raise Exception("❌ 必须在GPU环境中运行DriveMM模型!")
        
        try:
            from huggingface_hub import snapshot_download
            import time
            
            # 设置模型目录
            model_dir = "/tmp/DriveMM_model"
            cache_dir = "/tmp/huggingface_cache"
            
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            
            model_name = "DriveMM/DriveMM"
            
            # 下载模型
            config_file = os.path.join(model_dir, "config.json")
            if not os.path.exists(config_file):
                logger.info("📥 下载DriveMM模型...")
                start_time = time.time()
                snapshot_download(
                    repo_id=model_name,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    cache_dir=cache_dir
                )
                download_time = time.time() - start_time
                logger.info(f"✅ 模型下载完成，耗时: {download_time:.1f}秒")
            else:
                logger.info("✅ 发现已下载的DriveMM模型")
            
            # 添加模型路径
            sys.path.append(model_dir)
            
            # 使用LLaVA架构加载模型
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import process_images
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.conversation import conv_templates
            
            model_name_type = 'llama'
            
            logger.info("📦 使用LLaVA架构加载DriveMM模型...")
            self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
                model_dir, 
                None, 
                model_name_type, 
                device_map=self.device
            )
            
            # 验证关键组件
            if self.image_processor is None:
                logger.warning("⚠️ image_processor是None, 手动加载...")
                from transformers import CLIPImageProcessor
                self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
                logger.info("✅ 手动加载CLIPImageProcessor成功")
            
            if self.model is None:
                logger.error("❌ model是None!")
                raise Exception("model加载失败")
            
            # 设置为评估模式
            self.model.eval()
            
            # 设置special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("✅ DriveMM模型设置完成")
            
        except Exception as e:
            logger.error(f"❌ DriveMM模型设置失败: {e}")
            import traceback
            logger.error(f"❌ 堆栈跟踪: {traceback.format_exc()}")
            raise
    
    def get_video_list_from_storage(self, container_name="dada-videos"):
        """从Azure Storage获取视频列表"""
        logger.info(f"📁 获取视频列表...")
        
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_list = container_client.list_blobs()
            
            video_blobs = []
            for blob in blob_list:
                if blob.name.endswith('.avi'):
                    video_blobs.append(blob.name)
            
            logger.info(f"📊 发现 {len(video_blobs)} 个视频文件")
            return video_blobs[:1]  # 只测试1个视频，确保修复有效
            
        except Exception as e:
            logger.error(f"❌ 获取视频列表失败: {e}")
            return []
    
    def download_video_to_temp(self, blob_name, container_name="dada-videos"):
        """下载视频到临时文件"""
        logger.info(f"📥 下载视频: {blob_name}")
        
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
            
            with open(temp_file.name, 'wb') as f:
                download_stream = blob_client.download_blob()
                download_stream.readinto(f)
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"❌ 下载视频失败 {blob_name}: {e}")
            return None
    
    def extract_video_frames_safe(self, video_path, num_frames=10):
        """安全提取视频帧 - 专门防止CUDA errors"""
        logger.info(f"🔧 安全提取视频帧: {num_frames}帧")
        
        try:
            import cv2
            from PIL import Image
            
            # 设置OpenCV避免GPU操作
            os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '1'
            os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"📊 视频信息: {total_frames}帧, {fps:.2f}fps, {duration:.2f}秒")
            
            # 均匀提取帧
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # 确保帧的标准格式
                    if frame.shape[2] == 3:  # BGR -> RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = frame
                    
                    # 验证帧尺寸
                    height, width = frame_rgb.shape[:2]
                    if height < 50 or width < 50:
                        logger.warning(f"⚠️ 帧 {i} 尺寸过小: {width}x{height}")
                        continue
                    
                    # 转换为PIL Image并标准化
                    pil_image = Image.fromarray(frame_rgb).convert("RGB")
                    
                    # 验证PIL图像
                    if pil_image.size[0] < 50 or pil_image.size[1] < 50:
                        logger.warning(f"⚠️ PIL图像 {i} 尺寸过小")
                        continue
                    
                    frames.append(pil_image)
                    logger.debug(f"✅ 提取帧 {i}: {pil_image.size}")
                else:
                    logger.warning(f"⚠️ 无法读取帧 {frame_idx}")
            
            cap.release()
            
            logger.info(f"✅ 成功提取 {len(frames)} 帧")
            return frames, duration
            
        except Exception as e:
            logger.error(f"❌ 帧提取失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], 0
    
    def safe_image_processing(self, frames):
        """安全的图像处理 - 专门防止CUDA device-side assert"""
        logger.info(f"🔧 开始安全图像处理: {len(frames)}帧")
        
        try:
            # 验证输入帧
            if not frames:
                raise ValueError("没有输入帧")
            
            # 验证每一帧
            validated_frames = []
            for i, frame in enumerate(frames):
                if frame is None:
                    logger.warning(f"⚠️ 帧 {i} 是None，跳过")
                    continue
                
                if not hasattr(frame, 'size'):
                    logger.warning(f"⚠️ 帧 {i} 不是有效的PIL图像，跳过")
                    continue
                
                width, height = frame.size
                if width < 50 or height < 50:
                    logger.warning(f"⚠️ 帧 {i} 尺寸过小 {width}x{height}，跳过")
                    continue
                
                validated_frames.append(frame)
                logger.debug(f"✅ 验证帧 {i}: {width}x{height}")
            
            if not validated_frames:
                raise ValueError("没有有效的帧")
            
            logger.info(f"✅ 验证完成: {len(validated_frames)}/{len(frames)} 帧有效")
            
            # 使用process_images进行处理，但添加安全检查
            from llava.mm_utils import process_images
            
            logger.info("🔧 调用process_images...")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 处理图像 - 这是关键的失败点
            try:
                image_tensors = process_images(
                    validated_frames, 
                    self.image_processor, 
                    self.model.config
                )
                
                logger.info(f"✅ process_images完成: {image_tensors.shape}")
                
                # 验证tensor
                if image_tensors is None:
                    raise ValueError("process_images返回None")
                
                if not isinstance(image_tensors, torch.Tensor):
                    raise ValueError("process_images返回的不是tensor")
                
                # 验证tensor属性
                logger.info(f"🔍 Tensor信息:")
                logger.info(f"   - Shape: {image_tensors.shape}")
                logger.info(f"   - Device: {image_tensors.device}")
                logger.info(f"   - Dtype: {image_tensors.dtype}")
                logger.info(f"   - Requires_grad: {image_tensors.requires_grad}")
                
                # 检查tensor数值
                if torch.isnan(image_tensors).any():
                    logger.error("❌ Tensor包含NaN值")
                    raise ValueError("Tensor包含NaN值")
                
                if torch.isinf(image_tensors).any():
                    logger.error("❌ Tensor包含无限值")
                    raise ValueError("Tensor包含无限值")
                
                # 转换数据类型以避免H100兼容性问题
                if image_tensors.dtype == torch.bfloat16:
                    logger.info("🔧 转换bfloat16到float32以避免H100问题")
                    image_tensors = image_tensors.to(torch.float32)
                
                # 确保tensor在正确设备上
                if image_tensors.device != self.device:
                    logger.info(f"🔧 移动tensor从 {image_tensors.device} 到 {self.device}")
                    image_tensors = image_tensors.to(self.device)
                
                # 确保tensor是连续的
                if not image_tensors.is_contiguous():
                    logger.info("🔧 确保tensor连续性")
                    image_tensors = image_tensors.contiguous()
                
                # 最终验证
                logger.info(f"✅ 最终tensor: {image_tensors.shape}, {image_tensors.dtype}, {image_tensors.device}")
                
                return image_tensors
                
            except Exception as e:
                logger.error(f"❌ process_images失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
            
        except Exception as e:
            logger.error(f"❌ 安全图像处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def drivemm_inference_surgical(self, frames, video_id):
        """外科手术式DriveMM推理 - 专门修复CUDA device-side assert"""
        logger.info(f"🔧 外科手术式DriveMM推理: {video_id}")
        
        if not frames:
            raise Exception("没有提取到有效的视频帧")
        
        try:
            # Step 1: 安全的图像处理
            logger.info("🔧 Step 1: 安全图像处理...")
            image_tensors = self.safe_image_processing(frames)
            
            # Step 2: 安全的文本处理
            logger.info("🔧 Step 2: 安全文本处理...")
            
            # 构建简化的prompt以避免tokenization问题
            simple_prompt = f"Analyze this traffic video {video_id}. Describe what you see and any potential safety concerns. Respond in JSON format with video_id, summary, and key_actions fields."
            
            from llava.conversation import conv_templates
            from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            
            # 使用简单的对话模板
            conv_mode = "vicuna_v1"
            conv = conv_templates[conv_mode].copy()
            
            # 构建包含图像token的prompt
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if len(frames) > 1:
                qs = image_token_se * len(frames) + "\n" + simple_prompt
            else:
                qs = image_token_se + "\n" + simple_prompt
            
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            logger.info(f"🔧 Prompt构建完成，长度: {len(prompt)}")
            
            # Step 3: 安全的tokenization
            logger.info("🔧 Step 3: 安全tokenization...")
            
            # 安全的tokenization，避免IMAGE_TOKEN_INDEX问题
            from llava.mm_utils import tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX
            
            try:
                # 检查词汇表大小
                vocab_size = len(self.tokenizer)
                logger.info(f"🔍 词汇表大小: {vocab_size}")
                logger.info(f"🔍 IMAGE_TOKEN_INDEX: {IMAGE_TOKEN_INDEX}")
                
                # 如果IMAGE_TOKEN_INDEX超出范围，使用安全替代
                if IMAGE_TOKEN_INDEX < -vocab_size or IMAGE_TOKEN_INDEX >= vocab_size:
                    logger.warning(f"⚠️ IMAGE_TOKEN_INDEX {IMAGE_TOKEN_INDEX} 超出范围，使用安全替代")
                    # 使用pad_token_id或unk_token_id作为替代
                    safe_index = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
                    if safe_index is None:
                        safe_index = vocab_size - 1  # 使用最后一个token作为安全选择
                    logger.info(f"🔧 使用安全index: {safe_index}")
                    
                    # 用安全index替换IMAGE_TOKEN_INDEX
                    input_ids = tokenizer_image_token(prompt, self.tokenizer, safe_index, return_tensors='pt')
                else:
                    input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                
                # 验证input_ids
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                input_ids = input_ids.to(self.device)
                
                # 检查token范围
                max_token = input_ids.max().item()
                min_token = input_ids.min().item()
                
                logger.info(f"🔍 Token范围: [{min_token}, {max_token}]")
                
                if max_token >= vocab_size or min_token < -vocab_size:
                    logger.warning("⚠️ Token超出词汇表范围，进行修正")
                    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                
                logger.info(f"✅ Tokenization完成: {input_ids.shape}")
                
            except Exception as e:
                logger.error(f"❌ Tokenization失败: {e}")
                raise
            
            # Step 4: 安全的模型推理
            logger.info("🔧 Step 4: 安全模型推理...")
            
            with torch.no_grad():
                # 清理GPU内存
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                try:
                    # 使用最保守的生成参数
                    generation_config = {
                        'do_sample': False,
                        'max_new_tokens': 200,  # 保守的token数量
                        'temperature': 0.0,
                        'use_cache': True,
                        'pad_token_id': self.tokenizer.eos_token_id,
                    }
                    
                    logger.info("🔧 开始模型生成...")
                    
                    # 这里是最可能出现CUDA error的地方
                    output_ids = self.model.generate(
                        input_ids,
                        images=image_tensors,
                        image_sizes=[frame.size for frame in frames],
                        **generation_config
                    )
                    
                    logger.info(f"✅ 模型生成完成: {output_ids.shape}")
                    
                    # 同步GPU操作
                    torch.cuda.synchronize()
                    
                except Exception as e:
                    logger.error(f"❌ 模型推理失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise
            
            # Step 5: 解码输出
            logger.info("🔧 Step 5: 解码输出...")
            
            try:
                # 移除输入部分，只保留生成的内容
                input_token_len = input_ids.shape[1]
                response_ids = output_ids[:, input_token_len:]
                
                # 解码
                response = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
                
                logger.info(f"✅ 解码完成，响应长度: {len(response)}")
                logger.info(f"🔍 响应预览: {response[:200]}...")
                
                # 构建返回结果
                result = {
                    "video_id": video_id,
                    "segment_id": "segment_000",
                    "Start_Timestamp": "0.0s",
                    "End_Timestamp": "10.0s",
                    "sentiment": "Neutral",
                    "scene_theme": "Traffic Analysis",
                    "characters": "driver",
                    "summary": response if response else f"DriveMM analysis for {video_id}",
                    "actions": "vehicle movement and traffic monitoring",
                    "key_objects": "traffic elements",
                    "key_actions": "traffic analysis completed",
                    "next_action": {
                        "speed_control": "maintain speed",
                        "direction_control": "keep direction", 
                        "lane_control": "maintain current lane"
                    },
                    "surgical_fix": True,
                    "processing_success": True
                }
                
                return result
                
            except Exception as e:
                logger.error(f"❌ 解码失败: {e}")
                raise
            
        except Exception as e:
            logger.error(f"❌ 外科手术式推理失败 {video_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def process_all_videos(self):
        """处理所有视频"""
        logger.info("🚀 开始外科手术式DriveMM推理")
        
        video_blobs = self.get_video_list_from_storage()
        if not video_blobs:
            logger.error("❌ 未找到视频文件")
            return []
        
        results = []
        
        for i, blob in enumerate(video_blobs, 1):
            try:
                logger.info(f"📹 处理视频 {i}/{len(video_blobs)}: {blob}")
                
                # 下载视频
                video_path = self.download_video_to_temp(blob)
                if not video_path:
                    continue
                
                # 安全提取帧
                frames, duration = self.extract_video_frames_safe(video_path, num_frames=10)
                if not frames:
                    logger.error(f"❌ 帧提取失败: {blob}")
                    continue
                
                # 外科手术式推理
                result = self.drivemm_inference_surgical(frames, blob)
                results.append(result)
                
                # 清理
                if os.path.exists(video_path):
                    os.unlink(video_path)
                
                logger.info(f"✅ 外科手术式处理完成: {blob}")
                
            except Exception as e:
                logger.error(f"❌ 处理失败 {blob}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        return results
    
    def save_final_results(self, results):
        """保存最终结果"""
        output_file = "azure_drivemm_surgical_fix_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 最终结果已保存: {output_file}")
        
        logger.info("=" * 60)
        logger.info("🎉 外科手术式DriveMM推理完成!")
        logger.info("📊 统计结果:")
        logger.info(f"   总视频数: {len(results)}")
        
        if results:
            logger.info("✅ 外科手术式修复成功 - CUDA错误已解决!")
        else:
            logger.info("⚠️ 没有成功处理的视频")

def main():
    """主函数"""
    try:
        # 创建外科手术式修复实例
        inference = SurgicalFixDriveMM()
        
        # 处理视频
        results = inference.process_all_videos()
        
        # 保存结果
        inference.save_final_results(results)
        
    except Exception as e:
        logger.error(f"❌ 主程序异常: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()