#!/usr/bin/env python3
"""
Run 11 Temperature修正实验 - 尝试复现GPT-4.1历史最佳结果
基于发现历史配置使用temperature=0.3而非0的关键差异
"""

import os
import sys
import json
import time
import base64
import logging
import requests
import pandas as pd
import cv2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip

# 添加项目根目录到Python路径
sys.path.append("/Users/wanmeng/repository/GPT4Video-cobra-auto")
import video_utilities as vu

# 加载环境变量
load_dotenv()

class GPT41TemperatureFixExperiment:
    def __init__(self):
        """初始化实验配置"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run11-gpt41-balanced-100videos"
        
        # API配置 - 修正关键的temperature参数
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.vision_endpoint = os.environ["VISION_ENDPOINT"]
        if not self.vision_endpoint.endswith("/"):
            self.vision_endpoint += "/"
        self.vision_deployment = os.environ.get("GPT_4.1_VISION_DEPLOYMENT_NAME", "gpt-4.1")
        
        # 关键修正：使用历史最佳配置的temperature=0.3
        self.temperature = 0.3  # 历史配置使用0.3，Run 11使用0
        
        # 其他历史匹配配置
        self.max_tokens = 2000
        self.api_version = "2024-02-15-preview"
        
        # 视频处理配置
        self.frame_interval = 10
        self.frames_per_interval = 10
        self.fps = 1
        
        # 数据集配置
        self.video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos"
        self.ground_truth_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, f'temperature_fix_experiment_{self.timestamp}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 加载ground truth数据
        self.ground_truth = pd.read_csv(self.ground_truth_path, sep='\t', engine='python')
        # 处理标签，将"ghost probing"转换为1，"none"转换为0
        self.ground_truth['ghost_probing'] = self.ground_truth['ground_truth_label'].apply(
            lambda x: 1 if 'ghost probing' in str(x) else 0
        )
        self.logger.info(f"加载ground truth数据: {len(self.ground_truth)}个视频")
        
        # 结果存储
        self.results = {
            "experiment_info": {
                "run_id": "Run 11 Temperature Fix",
                "timestamp": self.timestamp,
                "key_change": "temperature: 0 -> 0.3 (历史配置恢复)",
                "model": "GPT-4.1 (Azure)",
                "prompt_version": "Balanced (Historical Best Recreation)",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "purpose": "修正temperature参数以复现历史最佳结果",
                "target_metrics": {
                    "f1_score": 0.712,
                    "recall": 0.963,
                    "precision": 0.565
                }
            },
            "detailed_results": []
        }
        
        self.logger.info("🧪 Temperature修正实验初始化完成")
        self.logger.info(f"🔥 关键修正: temperature = {self.temperature} (历史配置)")
        self.logger.info(f"📊 目标: F1=0.712, 召回率=0.963, 精确度=0.565")
    
    def get_balanced_prompt(self, video_id, segment_id="full_video", start_time=0, end_time=10):
        """获取历史最佳的Balanced prompt - 完全一致版本"""
        return f"""You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {self.frame_interval} seconds of audio from a video,
as well as {self.frames_per_interval} frames split evenly throughout {self.frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({self.frames_per_interval}
frames over {self.frame_interval} seconds), which is generated from your analysis of each frame ({self.frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the video.

IMPORTANT: For ghost probing detection, consider TWO categories:

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

Your response should be a valid JSON object with the following EXACT structure (match this format precisely):
{{
    "video_id": "{video_id}",
    "segment_id": "{segment_id}",
    "Start_Timestamp": "{start_time:.1f}s",
    "End_Timestamp": "{end_time:.1f}s",
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

Audio Transcription: [No audio analysis in this experiment]"""
    
    def send_azure_openai_request(self, prompt, frames):
        """发送Azure OpenAI请求 - 使用修正的temperature参数"""
        encoded_images = []
        for frame_path in frames:
            with open(frame_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append(encoded_string)
        
        content = [{"type": "text", "text": prompt}]
        
        for encoded_image in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })
        
        # 关键修正：使用temperature=0.3而非0
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature  # 🔥 关键修正点
        }
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.openai_api_key
        }
        
        url = f"{self.vision_endpoint}openai/deployments/{self.vision_deployment}/chat/completions?api-version={self.api_version}"
        
        # 历史配置的重试机制
        max_retries = 2
        wait_exponential_multiplier = 2000
        wait_exponential_max = 60000
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=90)
                
                if response.status_code == 200:
                    response_data = response.json()
                    return response_data['choices'][0]['message']['content']
                else:
                    self.logger.error(f"API请求失败 (尝试 {attempt + 1}/{max_retries}): {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout as e:
                self.logger.error(f"API请求超时 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            except Exception as e:
                self.logger.error(f"API请求异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = min(wait_exponential_multiplier * (2 ** attempt), wait_exponential_max) / 1000
                time.sleep(wait_time)
        
        return None
    
    def extract_frames_from_video(self, video_path):
        """从视频中提取帧 - 自实现版本"""
        frames_dir = "frames_temp"
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        
        try:
            # 使用moviepy提取帧
            video_clip = VideoFileClip(video_path)
            duration = video_clip.duration
            
            frame_files = []
            for i in range(self.frames_per_interval):
                frame_time = i * (self.frame_interval / self.frames_per_interval)
                if frame_time >= duration:
                    break
                
                frame_path = os.path.join(frames_dir, f"frame_at_{frame_time:.1f}s.jpg")
                
                # 提取帧并保存
                frame = video_clip.get_frame(frame_time)
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_files.append(frame_path)
            
            video_clip.close()
            return frame_files
            
        except Exception as e:
            self.logger.error(f"帧提取失败: {str(e)}")
            return []
    
    def process_single_video(self, video_id):
        """处理单个视频"""
        start_time = time.time()
        
        # 获取ground truth标签
        gt_row = self.ground_truth[self.ground_truth['video_id'] == video_id]
        if gt_row.empty:
            self.logger.error(f"视频 {video_id} 在ground truth中未找到")
            return None
        
        actual_label = int(gt_row.iloc[0]['ghost_probing'])
        
        # 构建视频文件路径 - 处理重复扩展名问题
        if video_id.endswith('.avi'):
            video_path = os.path.join(self.video_dir, video_id)
        else:
            video_path = os.path.join(self.video_dir, f"{video_id}.avi")
        if not os.path.exists(video_path):
            self.logger.error(f"视频文件不存在: {video_path}")
            return None
        
        # 提取帧
        frames = self.extract_frames_from_video(video_path)
        if not frames:
            self.logger.error(f"视频 {video_id} 帧提取失败")
            return None
        
        # 生成prompt
        prompt = self.get_balanced_prompt(video_id)
        
        # 发送API请求
        response = self.send_azure_openai_request(prompt, frames)
        
        # 清理临时帧文件
        for frame_path in frames:
            if os.path.exists(frame_path):
                os.remove(frame_path)
        
        if response is None:
            self.logger.error(f"视频 {video_id} API请求失败")
            return {
                "video_id": video_id,
                "status": "api_error", 
                "actual_label": actual_label,
                "predicted_label": None,
                "processing_time": time.time() - start_time
            }
        
        # 解析响应
        try:
            parsed_result = json.loads(response)
            key_actions = parsed_result.get('key_actions', '').lower()
            
            # 判断是否为ghost probing
            predicted_label = 1 if 'ghost probing' in key_actions else 0
            
            processing_time = time.time() - start_time
            self.logger.info(f"视频 {video_id} 完成 - 预测: {predicted_label}, 实际: {actual_label}, 用时: {processing_time:.1f}s")
            
            return {
                "video_id": video_id,
                "status": "success",
                "actual_label": actual_label, 
                "predicted_label": predicted_label,
                "processing_time": processing_time,
                "raw_response": response,
                "parsed_result": parsed_result,
                "key_actions": key_actions
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"视频 {video_id} JSON解析失败: {str(e)}")
            return {
                "video_id": video_id,
                "status": "parse_error",
                "actual_label": actual_label,
                "predicted_label": None,
                "processing_time": time.time() - start_time,
                "raw_response": response
            }
    
    def calculate_metrics(self):
        """计算性能指标"""
        tp = fp = tn = fn = errors = 0
        
        for result in self.results["detailed_results"]:
            if result["status"] != "success":
                errors += 1
                continue
                
            predicted = result["predicted_label"]
            actual = result["actual_label"]
            
            if predicted == 1 and actual == 1:
                tp += 1
            elif predicted == 1 and actual == 0:
                fp += 1
            elif predicted == 0 and actual == 1:
                fn += 1
            else:
                tn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "errors": errors
        }
    
    def run_experiment(self, limit=20):
        """运行Temperature修正实验"""
        self.logger.info(f"🚀 开始Temperature修正实验 (limit={limit})")
        self.logger.info(f"🔥 使用temperature={self.temperature} (历史配置)")
        
        # 获取视频列表
        video_ids = self.ground_truth['video_id'].unique()[:limit]
        
        for i, video_id in enumerate(video_ids, 1):
            self.logger.info(f"处理视频 {i}/{len(video_ids)}: {video_id}")
            
            result = self.process_single_video(video_id)
            if result:
                self.results["detailed_results"].append(result)
            
            # 每5个视频保存一次中间结果
            if i % 5 == 0:
                self.save_intermediate_results()
        
        # 计算最终指标
        metrics = self.calculate_metrics()
        self.results["performance_metrics"] = metrics
        
        # 保存最终结果
        self.save_final_results()
        
        # 输出结果摘要
        self.logger.info("🏁 Temperature修正实验完成")
        self.logger.info(f"性能指标: F1={metrics['f1_score']:.3f}, 召回率={metrics['recall']:.3f}, 精确度={metrics['precision']:.3f}")
        
        # 与历史目标对比
        f1_diff = metrics['f1_score'] - 0.712
        recall_diff = metrics['recall'] - 0.963
        precision_diff = metrics['precision'] - 0.565
        
        self.logger.info("📊 与历史目标对比:")
        self.logger.info(f"  F1分数: {metrics['f1_score']:.3f} vs 0.712 ({f1_diff:+.3f})")
        self.logger.info(f"  召回率: {metrics['recall']:.3f} vs 0.963 ({recall_diff:+.3f})")
        self.logger.info(f"  精确度: {metrics['precision']:.3f} vs 0.565 ({precision_diff:+.3f})")
        
        return metrics
    
    def save_intermediate_results(self):
        """保存中间结果"""
        filename = os.path.join(self.output_dir, f"temperature_fix_intermediate_{self.timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
    
    def save_final_results(self):
        """保存最终结果"""
        # 保存详细结果
        json_filename = os.path.join(self.output_dir, f"temperature_fix_final_results_{self.timestamp}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 保存性能报告
        metrics = self.results["performance_metrics"]
        report_filename = os.path.join(self.output_dir, f"temperature_fix_report_{self.timestamp}.md")
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"""# Temperature修正实验报告 (Run 11 Fix)

## 实验信息
- **运行时间**: {self.timestamp}
- **关键修正**: temperature: 0 → 0.3 (恢复历史配置)
- **模型**: GPT-4.1 (Azure)
- **处理视频数**: {len(self.results['detailed_results'])}个

## 性能结果

### 当前实验结果
- **F1分数**: {metrics['f1_score']:.3f}
- **召回率**: {metrics['recall']:.3f}
- **精确度**: {metrics['precision']:.3f}
- **准确率**: {metrics['accuracy']:.3f}

### 与历史目标对比
- **F1分数**: {metrics['f1_score']:.3f} vs 0.712 ({metrics['f1_score']-0.712:+.3f})
- **召回率**: {metrics['recall']:.3f} vs 0.963 ({metrics['recall']-0.963:+.3f})
- **精确度**: {metrics['precision']:.3f} vs 0.565 ({metrics['precision']-0.565:+.3f})

### 混淆矩阵
- **TP**: {metrics['tp']}, **FP**: {metrics['fp']}
- **TN**: {metrics['tn']}, **FN**: {metrics['fn']}
- **错误**: {metrics['errors']}

## 结论
{'✅ Temperature修正显著改善性能' if metrics['f1_score'] > 0.4 else '❌ Temperature修正未能有效改善性能'}

## 建议
{'继续扩展到更多视频验证' if metrics['f1_score'] > 0.5 else '考虑其他复现方案或采用混合策略'}
""")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPT-4.1 Temperature修正实验')
    parser.add_argument('--limit', type=int, default=20, help='处理视频数量限制')
    parser.add_argument('--temperature', type=float, default=0.3, help='温度参数')
    
    args = parser.parse_args()
    
    # 运行实验
    experiment = GPT41TemperatureFixExperiment()
    if args.temperature != 0.3:
        experiment.temperature = args.temperature
        experiment.logger.info(f"使用自定义温度参数: {args.temperature}")
    
    metrics = experiment.run_experiment(limit=args.limit)
    
    print("\n" + "="*60)
    print("🧪 TEMPERATURE修正实验完成")
    print("="*60)
    print(f"📊 F1分数: {metrics['f1_score']:.3f} (目标: 0.712)")
    print(f"📊 召回率: {metrics['recall']:.3f} (目标: 0.963)")
    print(f"📊 精确度: {metrics['precision']:.3f} (目标: 0.565)")
    print("="*60)
    
    if metrics['f1_score'] > 0.5:
        print("✅ 实验显示正面效果，建议扩展验证")
    else:
        print("❌ 修正效果有限，考虑其他复现方案")

if __name__ == "__main__":
    main()