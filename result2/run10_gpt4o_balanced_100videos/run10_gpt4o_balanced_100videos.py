#!/usr/bin/env python3
"""
Run 10: GPT-4o Balanced Version (100 Videos)
基于GPT-4.1 Balanced的prompt设计，使用GPT-4o模型进行100视频完整测试
与GPT-4.1 Balanced保持完全一致的prompt和参数配置
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
from collections import Counter

# 加载环境变量
load_dotenv()

class GPT4oRun10BalancedExperiment:
    def __init__(self, output_dir, chunk_size=10):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logging()
        self.setup_openai_api()
        self.load_ground_truth()
        self.initialize_results()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        log_filename = os.path.join(self.output_dir, f"run10_gpt4o_balanced_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Run 10: GPT-4o Balanced Version (100 Videos) 开始")
        
    def setup_openai_api(self):
        """设置OpenAI API"""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY未设置")
        
        # Azure OpenAI配置
        self.vision_endpoint = os.environ.get("VISION_ENDPOINT", "")
        self.vision_deployment = os.environ.get("VISION_DEPLOYMENT_NAME", "gpt-4o-global")
        
        if not self.vision_endpoint:
            raise ValueError("VISION_ENDPOINT未设置")
            
        self.logger.info(f"Azure OpenAI API配置成功")
        self.logger.info(f"Endpoint: {self.vision_endpoint}")
        self.logger.info(f"Deployment: {self.vision_deployment}")
        self.logger.info(f"Temperature: 0.3 (与GPT-4.1 Balanced保持一致)")
        
    def load_ground_truth(self):
        """加载ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
        self.ground_truth = pd.read_csv(gt_path, sep='\t')
        self.logger.info(f"加载ground truth标签: {len(self.ground_truth)}个视频")
        
    def initialize_results(self):
        """初始化结果结构"""
        self.results = {
            "experiment_info": {
                "run_id": "Run 10",
                "timestamp": self.timestamp,
                "video_count": 100,
                "model": "GPT-4o (Azure)",
                "prompt_version": "GPT-4.1 Balanced (移植到GPT-4o)",
                "temperature": 0.3,
                "max_tokens": 2000,
                "purpose": "对比GPT-4o在Balanced prompt下的100视频性能",
                "output_directory": self.output_dir,
                "prompt_characteristics": [
                    "三层ghost probing分类系统",
                    "环境上下文整合",
                    "平衡精确度与召回率",
                    "简化验证流程",
                    "与GPT-4.1 Balanced完全一致的prompt",
                    "Temperature=0.3保持历史一致性"
                ]
            },
            "detailed_results": []
        }
        
    def extract_frames_from_video(self, video_path, frame_interval=10, frames_per_interval=10):
        """从视频中提取帧"""
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            frames = []
            frames_dir = os.path.join(self.output_dir, "frames_temp")
            os.makedirs(frames_dir, exist_ok=True)
            
            # 计算间隔数
            num_intervals = max(1, int(duration / frame_interval))
            
            for interval_idx in range(num_intervals):
                start_time = interval_idx * frame_interval
                end_time = min((interval_idx + 1) * frame_interval, duration)
                
                for frame_idx in range(frames_per_interval):
                    if frames_per_interval == 1:
                        frame_time = start_time + (end_time - start_time) / 2
                    else:
                        frame_time = start_time + (frame_idx / (frames_per_interval - 1)) * (end_time - start_time)
                    
                    if frame_time >= duration:
                        break
                        
                    frame = clip.get_frame(frame_time)
                    frame_filename = f"frame_{interval_idx}_{frame_idx}_{frame_time:.1f}s.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frames.append(frame_path)
            
            clip.close()
            return frames
        except Exception as e:
            self.logger.error(f"帧提取失败 {video_path}: {str(e)}")
            return []
    
    def get_gpt4o_balanced_prompt(self, video_id, frame_interval=10, frames_per_interval=10):
        """获取与GPT-4.1 Balanced完全一致的prompt（移植到GPT-4o）"""
        return f'''You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the video.

IMPORTANT: For ghost probing detection, consider THREE categories:

**1. HIGH-CONFIDENCE Ghost Probing (use "ghost probing" in key_actions)**:
- Object appears EXTREMELY close (within 1-2 vehicle lengths, <3 meters) 
- Appearance is SUDDEN and from blind spots (behind parked cars, buildings, corners)
- Occurs in HIGH-RISK environments: highways, rural roads, parking lots, uncontrolled intersections
- Requires IMMEDIATE emergency braking/swerving to avoid collision
- Movement is COMPLETELY UNPREDICTABLE and violates traffic expectations

**2. POTENTIAL Ghost Probing (use "potential ghost probing" in key_actions)**:
- Object appears suddenly but at moderate distance (3-5 meters)
- Sudden movement in environments where some unpredictability exists
- Requires emergency braking but collision risk is moderate
- Movement is unexpected but not completely impossible given the context

**3. NORMAL Traffic Situations (do NOT use "ghost probing")**:
- Pedestrians crossing at intersections, crosswalks, or traffic lights
- Vehicles making normal lane changes, turns, or merging with signals
- Cyclists following predictable paths in urban areas or bike lanes
- Any movement that is EXPECTED given the traffic environment and context

**Environment Context Guidelines**:
- INTERSECTION/CROSSWALK: Expect pedestrians and cyclists - use "emergency braking due to pedestrian crossing"
- HIGHWAY/RURAL: Higher chance of genuine ghost probing - be more sensitive
- PARKING LOT: Expect sudden vehicle movements - use "potential ghost probing" if very sudden
- URBAN STREET: Mixed - consider visibility and predictability

Use "ghost probing" for clear cases, "potential ghost probing" for borderline cases, and descriptive terms for normal traffic situations.

Your response should be a valid JSON object with the following EXACT structure:
{{
    "video_id": "{video_id}",
    "segment_id": "full_video",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "{frame_interval}.0s",
    "sentiment": "Positive/Negative/Neutral",
    "scene_theme": "Dramatic/Routine/Dangerous/Safe",
    "characters": "brief description of people in the scene",
    "summary": "comprehensive summary of the scene and what happens",
    "actions": "actions taken by the vehicle and driver responses",
    "key_objects": "numbered list: 1) Position: object description, distance, behavior impact 2) Position: object description, distance, behavior impact",
    "key_actions": "brief description of most important actions (use 'ghost probing', 'potential ghost probing', or descriptive terms as appropriate)",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}

Audio Transcription: [No audio available for this analysis]

Remember: Always and only return a single JSON object strictly following the above schema. Use the three-tier classification system exactly as specified to achieve optimal balance between precision and recall.'''
    
    def send_azure_openai_request(self, prompt, images):
        """发送Azure OpenAI请求 - 使用Temperature=0.3保持与GPT-4.1 Balanced一致"""
        encoded_images = []
        for image_path in images:
            try:
                with open(image_path, 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    encoded_images.append(encoded_string)
            except Exception as e:
                self.logger.error(f"图像编码失败 {image_path}: {str(e)}")
                continue
        
        if not encoded_images:
            return None
            
        content = [{"type": "text", "text": prompt}]
        
        for encoded_image in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })
        
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 2000,  # 与GPT-4.1 Balanced保持一致
            "temperature": 0.3   # 与GPT-4.1 Balanced保持一致
        }
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.openai_api_key
        }
        
        try:
            response = requests.post(
                f"{self.vision_endpoint}/openai/deployments/{self.vision_deployment}/chat/completions?api-version=2024-02-01",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            self.logger.error(f"API调用失败: {str(e)}")
            return None
    
    def analyze_with_gpt4o(self, video_path, video_id):
        """使用GPT-4o分析视频（Balanced prompt）"""
        try:
            # 提取帧
            frames = self.extract_frames_from_video(video_path)
            if not frames:
                return None
            
            # 生成prompt
            prompt = self.get_gpt4o_balanced_prompt(video_id)
            
            # 发送API请求
            result = self.send_azure_openai_request(prompt, frames)
            
            # 清理临时帧文件
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            
            return result
        except Exception as e:
            self.logger.error(f"视频分析失败 {video_id}: {str(e)}")
            return None
    
    def extract_key_actions(self, result_text):
        """提取key_actions"""
        try:
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            result_json = json.loads(result_text)
            return result_json.get('key_actions', '').lower()
        except:
            # 尝试正则表达式提取
            key_actions_match = re.search(r'"key_actions":\s*"([^"]*)"', result_text)
            if key_actions_match:
                return key_actions_match.group(1).lower()
            return result_text.lower()
    
    def evaluate_result(self, video_id, key_actions, ground_truth_label):
        """评估结果"""
        has_ghost_probing = ("ghost probing" in key_actions) or ("potential ghost probing" in key_actions)
        ground_truth_has_ghost = ground_truth_label != "none"
        
        if has_ghost_probing and ground_truth_has_ghost:
            return "TP"
        elif has_ghost_probing and not ground_truth_has_ghost:
            return "FP"
        elif not has_ghost_probing and ground_truth_has_ghost:
            return "FN"
        else:
            return "TN"
    
    def run_experiment(self):
        """运行Run 10 GPT-4o Balanced实验"""
        # 从ground truth文件中获取完整的100个视频列表
        test_videos = self.ground_truth['video_id'].tolist()
        
        self.logger.info(f"开始Run 10实验，处理 {len(test_videos)} 个视频")
        self.logger.info(f"使用GPT-4.1 Balanced相同的prompt设计")
        
        start_time = time.time()
        
        for i, video_id in enumerate(tqdm.tqdm(test_videos, desc="处理视频")):
            try:
                self.logger.info(f"处理视频 {i+1}/100: {video_id}")
                
                # 视频路径
                video_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/{video_id}"
                if not os.path.exists(video_path):
                    self.logger.warning(f"视频不存在: {video_path}")
                    continue
                
                # 获取ground truth
                gt_row = self.ground_truth[self.ground_truth['video_id'] == video_id]
                if gt_row.empty:
                    self.logger.warning(f"未找到ground truth: {video_id}")
                    continue
                
                ground_truth_label = gt_row.iloc[0]['ground_truth_label']
                
                # 分析视频
                result = self.analyze_with_gpt4o(video_path, video_id)
                
                if result:
                    key_actions = self.extract_key_actions(result)
                    evaluation = self.evaluate_result(video_id, key_actions, ground_truth_label)
                else:
                    key_actions = ""
                    evaluation = "ERROR"
                
                # 记录结果
                result_entry = {
                    "video_id": video_id,
                    "ground_truth": ground_truth_label,
                    "key_actions": key_actions,
                    "evaluation": evaluation,
                    "raw_result": result
                }
                
                self.results["detailed_results"].append(result_entry)
                
                self.logger.info(f"视频 {video_id}: GT={ground_truth_label}, 检测={key_actions}, 评估={evaluation}")
                
                # 每5个视频保存一次中间结果
                if (i + 1) % 5 == 0:
                    self.save_intermediate_results(i + 1)
                
            except Exception as e:
                self.logger.error(f"处理视频失败 {video_id}: {str(e)}")
                # 记录错误结果
                error_entry = {
                    "video_id": video_id,
                    "ground_truth": ground_truth_label if 'ground_truth_label' in locals() else "unknown",
                    "key_actions": "",
                    "evaluation": "ERROR",
                    "raw_result": f"处理错误: {str(e)}"
                }
                self.results["detailed_results"].append(error_entry)
                continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        self.logger.info(f"Run 10实验完成，总耗时: {total_time/60:.1f} 分钟")
        
        # 保存最终结果
        self.save_final_results()
        self.generate_performance_metrics()
        
    def save_intermediate_results(self, processed_count):
        """保存中间结果"""
        intermediate_file = os.path.join(self.output_dir, f"run10_intermediate_{processed_count}videos_{self.timestamp}.json")
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"中间结果已保存: {intermediate_file}")
    
    def save_final_results(self):
        """保存最终结果"""
        final_file = os.path.join(self.output_dir, f"run10_final_results_{self.timestamp}.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"最终结果已保存: {final_file}")
    
    def generate_performance_metrics(self):
        """生成性能指标并与GPT-4.1 Balanced对比"""
        from collections import Counter
        
        evaluations = [r['evaluation'] for r in self.results["detailed_results"]]
        eval_counts = Counter(evaluations)
        
        tp = eval_counts.get('TP', 0)
        fp = eval_counts.get('FP', 0)
        tn = eval_counts.get('TN', 0)
        fn = eval_counts.get('FN', 0)
        errors = eval_counts.get('ERROR', 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "errors": errors,
            "total_videos": len(self.results["detailed_results"]),
            "comparison_with_gpt41_balanced": {
                "gpt41_balanced_f1": 0.712,
                "gpt41_balanced_recall": 0.963,
                "gpt41_balanced_precision": 0.565,
                "gpt4o_balanced_f1": f1,
                "gpt4o_balanced_recall": recall,
                "gpt4o_balanced_precision": precision,
                "f1_difference": f1 - 0.712,
                "recall_difference": recall - 0.963,
                "precision_difference": precision - 0.565
            }
        }
        
        self.logger.info("=== Run 10 Performance Metrics (GPT-4o Balanced) ===")
        self.logger.info(f"精确度: {precision:.3f} ({precision*100:.1f}%)")
        self.logger.info(f"召回率: {recall:.3f} ({recall*100:.1f}%)")
        self.logger.info(f"F1分数: {f1:.3f} ({f1*100:.1f}%)")
        self.logger.info(f"准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
        self.logger.info(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, ERROR: {errors}")
        
        self.logger.info("=== 与GPT-4.1 Balanced对比 ===")
        self.logger.info(f"F1分数对比: GPT-4o={f1:.3f} vs GPT-4.1={0.712:.3f} (差异: {f1-0.712:+.3f})")
        self.logger.info(f"召回率对比: GPT-4o={recall:.3f} vs GPT-4.1={0.963:.3f} (差异: {recall-0.963:+.3f})")
        self.logger.info(f"精确度对比: GPT-4o={precision:.3f} vs GPT-4.1={0.565:.3f} (差异: {precision-0.565:+.3f})")
        
        # 保存指标
        metrics_file = os.path.join(self.output_dir, f"run10_metrics_{self.timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 创建输出目录
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run10_gpt4o_balanced_100videos"
    
    # 运行实验
    experiment = GPT4oRun10BalancedExperiment(output_dir)
    experiment.run_experiment()
    
    print("🎯 Run 10: GPT-4o Balanced Version (100 Videos) 实验完成!")
    print(f"📁 结果保存在: {output_dir}")
    print("📊 这将提供GPT-4o与GPT-4.1在相同Balanced prompt下的直接对比!")