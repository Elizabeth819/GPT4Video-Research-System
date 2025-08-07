#!/usr/bin/env python3
"""
Few-shot Ablation Experiment Runner
执行完整的消融实验：比较baseline、text few-shot和image few-shot三组
"""

import cv2
import os
import json
import logging
import time
import datetime
from moviepy.editor import VideoFileClip
import pandas as pd
from dotenv import load_dotenv
import tqdm
import re
import base64
import requests
import traceback
import sys
import glob
from fewshot_ablation_experiment import FewshotAblationExperiment

class AblationExperimentRunner(FewshotAblationExperiment):
    def __init__(self, video_limit=20):
        super().__init__()
        self.video_limit = video_limit
        self.frame_interval = 10  # 秒
        self.frames_per_interval = 10
        self.video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/DADA-200-videos"
        
    def extract_frames_from_video(self, video_path, output_dir, video_id):
        """从视频提取帧"""
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            # 创建临时帧目录
            frames_dir = os.path.join(output_dir, "frames_temp")
            os.makedirs(frames_dir, exist_ok=True)
            
            # 计算提取帧的时间点
            interval_count = max(1, int(duration // self.frame_interval))
            frame_paths = []
            
            for i in range(interval_count):
                start_time = i * self.frame_interval
                end_time = min((i + 1) * self.frame_interval, duration)
                
                # 在区间内均匀提取帧
                for j in range(self.frames_per_interval):
                    if interval_count == 1:
                        timestamp = start_time + (j * (end_time - start_time) / self.frames_per_interval)
                    else:
                        timestamp = start_time + (j * self.frame_interval / self.frames_per_interval)
                    
                    if timestamp < duration:
                        frame = clip.get_frame(timestamp)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        frame_filename = f"{video_id}_frame_{i:03d}_{j:02d}_{timestamp:.2f}s.jpg"
                        frame_path = os.path.join(frames_dir, frame_filename)
                        
                        if cv2.imwrite(frame_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            frame_paths.append(frame_path)
            
            clip.close()
            return frame_paths, duration
            
        except Exception as e:
            self.logger.error(f"视频帧提取失败 {video_path}: {str(e)}")
            return [], 0
    
    def send_azure_openai_request(self, prompt, image_paths):
        """发送Azure OpenAI请求"""
        try:
            # 编码图像
            encoded_images = []
            for image_path in image_paths:
                try:
                    with open(image_path, 'rb') as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        encoded_images.append(encoded_string)
                except Exception as e:
                    self.logger.error(f"图像编码失败 {image_path}: {str(e)}")
                    continue
            
            if not encoded_images:
                return None
                
            # 构建请求内容
            content = [{"type": "text", "text": prompt}]
            
            for encoded_image in encoded_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                        "detail": "high"
                    }
                })
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "api-key": self.openai_api_key
            }
            
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 3000,
                "temperature": 0
            }
            
            url = f"{self.vision_endpoint}/openai/deployments/{self.vision_deployment}/chat/completions?api-version=2023-12-01-preview"
            
            response = requests.post(url, headers=headers, json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                self.logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"API请求异常: {str(e)}")
            return None
    
    def process_single_video_group(self, video_path, group_name, prompt_type):
        """处理单个视频的特定实验组"""
        video_id = os.path.basename(video_path).replace('.avi', '')
        group_dir = self.group_dirs[group_name]
        
        self.logger.info(f"处理视频 {video_id} - 组: {group_name}")
        
        # 提取视频帧
        frame_paths, duration = self.extract_frames_from_video(video_path, group_dir, video_id)
        if not frame_paths:
            return None
        
        # 获取相应的prompt
        if prompt_type == "baseline":
            prompt = self.get_baseline_prompt(video_id, self.frame_interval)
            # 只使用视频帧
            request_images = frame_paths
        elif prompt_type == "text_fewshot":
            prompt = self.get_text_fewshot_prompt(video_id, self.frame_interval)
            # 只使用视频帧
            request_images = frame_paths
        elif prompt_type == "image_fewshot":
            prompt = self.get_image_fewshot_prompt(video_id, self.frame_interval)
            # 使用few-shot图像 + 视频帧
            request_images = self.fewshot_images + frame_paths
        else:
            self.logger.error(f"未知的prompt类型: {prompt_type}")
            return None
        
        # 发送API请求
        response = self.send_azure_openai_request(prompt, request_images)
        
        if response:
            try:
                # 尝试解析JSON
                json_response = json.loads(response)
                
                # 保存结果
                result_file = os.path.join(group_dir, f"actionSummary_{video_id}.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(json_response, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"成功处理 {video_id} - {group_name}")
                return json_response
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON解析失败 {video_id}: {str(e)}")
                # 保存原始响应用于调试
                raw_file = os.path.join(group_dir, f"raw_response_{video_id}.txt")
                with open(raw_file, 'w', encoding='utf-8') as f:
                    f.write(response)
                return None
        else:
            self.logger.error(f"API请求失败 {video_id} - {group_name}")
            return None
    
    def run_full_experiment(self):
        """运行完整的消融实验"""
        # 获取视频列表
        video_files = glob.glob(os.path.join(self.video_dir, "*.avi"))
        video_files = sorted(video_files)[:self.video_limit]
        
        self.logger.info(f"开始消融实验 - {len(video_files)}个视频")
        self.logger.info(f"实验组: {list(self.group_dirs.keys())}")
        
        # 初始化结果统计
        experiment_results = {
            "baseline": {"processed": 0, "successful": 0, "results": []},
            "text_fewshot": {"processed": 0, "successful": 0, "results": []},
            "image_fewshot": {"processed": 0, "successful": 0, "results": []}
        }
        
        # 逐个视频处理
        for video_path in tqdm.tqdm(video_files, desc="处理视频"):
            video_id = os.path.basename(video_path).replace('.avi', '')
            
            # 为每个实验组处理这个视频
            for group_name, prompt_type in [
                ("baseline", "baseline"),
                ("text_fewshot", "text_fewshot"), 
                ("image_fewshot", "image_fewshot")
            ]:
                experiment_results[group_name]["processed"] += 1
                
                result = self.process_single_video_group(video_path, group_name, prompt_type)
                
                if result:
                    experiment_results[group_name]["successful"] += 1
                    experiment_results[group_name]["results"].append({
                        "video_id": video_id,
                        "result": result
                    })
                
                # 短暂延迟避免API限制
                time.sleep(2)
        
        # 保存实验汇总
        summary_path = os.path.join(self.experiment_dir, "experiment_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info("消融实验完成")
        self.logger.info(f"结果汇总保存至: {summary_path}")
        
        # 打印统计信息
        for group_name, stats in experiment_results.items():
            success_rate = (stats["successful"] / stats["processed"]) * 100 if stats["processed"] > 0 else 0
            print(f"{group_name}: {stats['successful']}/{stats['processed']} ({success_rate:.1f}%)")
        
        return experiment_results

if __name__ == "__main__":
    # 运行消融实验
    runner = AblationExperimentRunner(video_limit=20)  # 先用20个视频测试
    results = runner.run_full_experiment()
    
    print("\n=== 消融实验完成 ===")
    print(f"实验目录: {runner.experiment_dir}")
    print("现在可以运行分析脚本进行性能对比")