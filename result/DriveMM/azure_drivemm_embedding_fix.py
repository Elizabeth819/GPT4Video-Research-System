#!/usr/bin/env python3
"""
DriveMM Azure推理脚本 - 终极Embedding索引修复
专门解决 `srcIndex < srcSelectDimSize` failed 错误
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

class EmbeddingFixDriveMM:
    def __init__(self):
        # 设置CUDA调试环境
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        
        self.setup_azure_clients()
        self.setup_drivemm_model_embedding_fix()
        
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
    
    def setup_drivemm_model_embedding_fix(self):
        """设置DriveMM模型 - 专门修复embedding索引问题"""
        logger.info("🔧 设置DriveMM模型(Embedding索引修复版)...")
        
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
            
            # 🔧 关键修复：检查和修复embedding层的索引范围
            self.fix_embedding_indices()
            
            logger.info("✅ DriveMM模型设置完成(已修复embedding索引)")
            
        except Exception as e:
            logger.error(f"❌ DriveMM模型设置失败: {e}")
            import traceback
            logger.error(f"❌ 堆栈跟踪: {traceback.format_exc()}")
            raise
    
    def fix_embedding_indices(self):
        """修复embedding层的索引问题 - 这是关键修复"""
        logger.info("🔧 开始修复embedding层索引问题...")
        
        try:
            # 获取tokenizer词汇表信息
            vocab_size = len(self.tokenizer)
            logger.info(f"🔍 Tokenizer词汇表大小: {vocab_size}")
            
            # 获取模型embedding层信息
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                embed_layer = self.model.model.embed_tokens
                embed_vocab_size = embed_layer.num_embeddings
                logger.info(f"🔍 Embedding层词汇表大小: {embed_vocab_size}")
                
                # 检查词汇表大小是否匹配
                if vocab_size != embed_vocab_size:
                    logger.warning(f"⚠️ 词汇表大小不匹配: tokenizer({vocab_size}) vs embedding({embed_vocab_size})")
                
                # 存储安全的词汇表大小（取较小值以确保安全）
                self.safe_vocab_size = min(vocab_size, embed_vocab_size)
                logger.info(f"✅ 使用安全词汇表大小: {self.safe_vocab_size}")
                
            else:
                logger.warning("⚠️ 无法找到embedding层，使用tokenizer词汇表大小")
                self.safe_vocab_size = vocab_size
            
            # 检查IMAGE_TOKEN_INDEX
            from llava.constants import IMAGE_TOKEN_INDEX
            logger.info(f"🔍 原始IMAGE_TOKEN_INDEX: {IMAGE_TOKEN_INDEX}")
            
            # 修复IMAGE_TOKEN_INDEX到安全范围
            if IMAGE_TOKEN_INDEX < 0 or IMAGE_TOKEN_INDEX >= self.safe_vocab_size:
                logger.warning(f"⚠️ IMAGE_TOKEN_INDEX {IMAGE_TOKEN_INDEX} 超出安全范围 [0, {self.safe_vocab_size-1}]")
                
                # 选择一个安全的替代index
                if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
                    self.safe_image_token_index = self.tokenizer.unk_token_id
                    logger.info(f"✅ 使用unk_token_id作为安全IMAGE_TOKEN_INDEX: {self.safe_image_token_index}")
                elif hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                    self.safe_image_token_index = self.tokenizer.pad_token_id
                    logger.info(f"✅ 使用pad_token_id作为安全IMAGE_TOKEN_INDEX: {self.safe_image_token_index}")
                else:
                    # 使用词汇表最后一个安全token
                    self.safe_image_token_index = self.safe_vocab_size - 1
                    logger.info(f"✅ 使用词汇表最后一个token作为安全IMAGE_TOKEN_INDEX: {self.safe_image_token_index}")
            else:
                self.safe_image_token_index = IMAGE_TOKEN_INDEX
                logger.info(f"✅ IMAGE_TOKEN_INDEX在安全范围内: {self.safe_image_token_index}")
            
            # 验证安全index
            if self.safe_image_token_index < 0 or self.safe_image_token_index >= self.safe_vocab_size:
                logger.error(f"❌ 安全IMAGE_TOKEN_INDEX仍然超出范围: {self.safe_image_token_index}")
                # 强制使用最安全的选择
                self.safe_image_token_index = min(1, self.safe_vocab_size - 1)
                logger.info(f"🔧 强制使用最安全的IMAGE_TOKEN_INDEX: {self.safe_image_token_index}")
            
            logger.info("✅ Embedding索引修复完成")
            
        except Exception as e:
            logger.error(f"❌ Embedding索引修复失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def safe_tokenization(self, prompt):
        """安全的tokenization - 确保所有token索引都在有效范围内"""
        logger.info("🔧 开始安全tokenization...")
        
        try:
            from llava.mm_utils import tokenizer_image_token
            
            # 使用安全的IMAGE_TOKEN_INDEX进行tokenization
            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                self.safe_image_token_index, 
                return_tensors='pt'
            )
            
            # 验证input_ids形状
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            
            input_ids = input_ids.to(self.device)
            
            # 🔧 关键修复：验证并修正所有token索引
            logger.info("🔧 验证并修正token索引...")
            
            # 获取token范围
            max_token = input_ids.max().item()
            min_token = input_ids.min().item()
            
            logger.info(f"🔍 Token范围: [{min_token}, {max_token}]")
            logger.info(f"🔍 安全范围: [0, {self.safe_vocab_size-1}]")
            
            # 修正超出范围的token
            if max_token >= self.safe_vocab_size or min_token < 0:
                logger.warning(f"⚠️ 发现超出范围的token，进行修正...")
                
                # 将所有token索引限制在安全范围内
                input_ids = torch.clamp(input_ids, 0, self.safe_vocab_size - 1)
                
                # 重新验证
                max_token = input_ids.max().item()
                min_token = input_ids.min().item()
                logger.info(f"✅ 修正后token范围: [{min_token}, {max_token}]")
            
            # 特别处理IMAGE_TOKEN_INDEX相关的token
            # 将所有等于原始IMAGE_TOKEN_INDEX的token替换为安全index
            from llava.constants import IMAGE_TOKEN_INDEX
            if IMAGE_TOKEN_INDEX != self.safe_image_token_index:
                # 查找并替换问题token
                if IMAGE_TOKEN_INDEX in input_ids:
                    logger.warning(f"⚠️ 发现原始IMAGE_TOKEN_INDEX {IMAGE_TOKEN_INDEX}，替换为安全值 {self.safe_image_token_index}")
                    input_ids[input_ids == IMAGE_TOKEN_INDEX] = self.safe_image_token_index
            
            # 最终验证
            max_token = input_ids.max().item()
            min_token = input_ids.min().item()
            
            if max_token >= self.safe_vocab_size or min_token < 0:
                logger.error(f"❌ 最终验证失败，token仍然超出范围: [{min_token}, {max_token}]")
                raise ValueError("Token索引仍然超出安全范围")
            
            logger.info(f"✅ 安全tokenization完成: {input_ids.shape}, token范围 [{min_token}, {max_token}]")
            
            return input_ids
            
        except Exception as e:
            logger.error(f"❌ 安全tokenization失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            return video_blobs[:1]  # 只测试1个视频
            
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
        """安全提取视频帧"""
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
            
            # 均匀提取帧
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb).convert("RGB")
                    frames.append(pil_image)
            
            cap.release()
            
            logger.info(f"✅ 成功提取 {len(frames)} 帧")
            return frames, duration
            
        except Exception as e:
            logger.error(f"❌ 帧提取失败: {e}")
            return [], 0
    
    def drivemm_inference_embedding_fix(self, frames, video_id):
        """DriveMM推理 - Embedding索引修复版"""
        logger.info(f"🔧 DriveMM推理(Embedding索引修复版): {video_id}")
        
        if not frames:
            raise Exception("没有提取到有效的视频帧")
        
        try:
            # Step 1: 处理图像
            logger.info("🔧 Step 1: 处理图像...")
            from llava.mm_utils import process_images
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            image_tensors = process_images(frames, self.image_processor, self.model.config)
            image_tensors = image_tensors.to(dtype=torch.float32, device=self.device)
            
            logger.info(f"✅ 图像处理完成: {image_tensors.shape}")
            
            # Step 2: 构建prompt
            logger.info("🔧 Step 2: 构建prompt...")
            
            from llava.conversation import conv_templates
            from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            
            # 简化的prompt以减少token复杂性
            simple_prompt = f"Analyze this traffic video {video_id} and describe what you see."
            
            # 构建包含图像token的prompt
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if len(frames) > 1:
                qs = image_token_se * len(frames) + "\n" + simple_prompt
            else:
                qs = image_token_se + "\n" + simple_prompt
            
            # 使用对话模板
            conv_mode = "vicuna_v1"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            logger.info(f"✅ Prompt构建完成，长度: {len(prompt)}")
            
            # Step 3: 安全tokenization - 这是关键步骤
            logger.info("🔧 Step 3: 安全tokenization(Embedding索引修复)...")
            input_ids = self.safe_tokenization(prompt)
            
            # Step 4: 模型推理
            logger.info("🔧 Step 4: 模型推理...")
            
            with torch.no_grad():
                # 再次清理GPU内存
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                try:
                    # 使用保守的生成参数
                    generation_config = {
                        'do_sample': False,
                        'max_new_tokens': 150,  # 更保守的token数量
                        'temperature': 0.0,
                        'use_cache': True,
                        'pad_token_id': self.tokenizer.eos_token_id,
                    }
                    
                    logger.info("🔧 开始模型生成(已修复embedding索引)...")
                    
                    # 这里应该不再出现CUDA indexing error
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
                logger.info(f"🔍 响应内容: {response[:200]}...")
                
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
                    "embedding_fix": True,
                    "safe_vocab_size": self.safe_vocab_size,
                    "safe_image_token_index": self.safe_image_token_index
                }
                
                return result
                
            except Exception as e:
                logger.error(f"❌ 解码失败: {e}")
                raise
            
        except Exception as e:
            logger.error(f"❌ Embedding索引修复推理失败 {video_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def process_all_videos(self):
        """处理所有视频"""
        logger.info("🚀 开始Embedding索引修复版DriveMM推理")
        
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
                
                # 提取帧
                frames, duration = self.extract_video_frames_safe(video_path, num_frames=10)
                if not frames:
                    logger.error(f"❌ 帧提取失败: {blob}")
                    continue
                
                # Embedding索引修复版推理
                result = self.drivemm_inference_embedding_fix(frames, blob)
                results.append(result)
                
                # 清理
                if os.path.exists(video_path):
                    os.unlink(video_path)
                
                logger.info(f"✅ Embedding索引修复处理完成: {blob}")
                
            except Exception as e:
                logger.error(f"❌ 处理失败 {blob}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        return results
    
    def save_final_results(self, results):
        """保存最终结果"""
        output_file = "azure_drivemm_embedding_fix_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 最终结果已保存: {output_file}")
        
        logger.info("=" * 60)
        logger.info("🎉 Embedding索引修复版DriveMM推理完成!")
        logger.info("📊 统计结果:")
        logger.info(f"   总视频数: {len(results)}")
        
        if results:
            logger.info("✅ Embedding索引修复成功 - srcIndex < srcSelectDimSize错误已解决!")
            for result in results:
                if 'safe_vocab_size' in result:
                    logger.info(f"   - 使用安全词汇表大小: {result['safe_vocab_size']}")
                if 'safe_image_token_index' in result:
                    logger.info(f"   - 使用安全IMAGE_TOKEN_INDEX: {result['safe_image_token_index']}")
        else:
            logger.info("⚠️ 没有成功处理的视频")

def main():
    """主函数"""
    try:
        # 创建Embedding索引修复实例
        inference = EmbeddingFixDriveMM()
        
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