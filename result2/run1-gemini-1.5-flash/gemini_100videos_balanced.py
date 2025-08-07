#!/usr/bin/env python3
"""
Gemini-1.5-flash 100视频分析 - 使用balanced版本prompt
结果保存到专门的目录中
"""

import cv2
import os
import json
import logging
import time
import datetime
from moviepy.editor import VideoFileClip
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import tqdm
import re

# 加载环境变量
load_dotenv()

class GeminiBalancedAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logging()
        self.setup_gemini_api()
        self.load_ground_truth()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.output_dir, f"gemini_balanced_100videos_log_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Gemini Balanced 100视频分析开始")
        
    def setup_gemini_api(self):
        """设置Gemini API"""
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            self.logger.info(f"Gemini API配置成功，使用API Key: {self.gemini_api_key[:20]}...")
        else:
            raise ValueError("GEMINI_API_KEY未设置")
            
    def load_ground_truth(self):
        """加载ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
        self.ground_truth = pd.read_csv(gt_path, sep='\t')
        self.logger.info(f"加载ground truth标签: {len(self.ground_truth)}个视频")
        
    def extract_frames_from_video(self, video_path, frame_interval=10, frames_per_interval=10):
        """从视频中提取帧"""
        frames_dir = os.path.join(self.output_dir, "frames_temp")
        os.makedirs(frames_dir, exist_ok=True)
        
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            frames = []
            timestamps = []
            
            # 计算间隔时间点
            num_intervals = max(1, int(duration / frame_interval))
            
            for interval_idx in range(num_intervals):
                start_time = interval_idx * frame_interval
                end_time = min((interval_idx + 1) * frame_interval, duration)
                
                # 在每个间隔内均匀提取帧
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
                    timestamps.append(frame_time)
            
            clip.close()
            return frames, timestamps
            
        except Exception as e:
            self.logger.error(f"帧提取失败 {video_path}: {str(e)}")
            return [], []
            
    def get_balanced_prompt(self, video_id, frame_interval=10, frames_per_interval=10):
        """获取balanced版本的GPT-4.1 prompt，适配Gemini格式"""
        return f"""You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

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
}}"""

    def analyze_with_gemini(self, video_path, video_id):
        """使用Gemini分析视频"""
        try:
            frames, timestamps = self.extract_frames_from_video(video_path)
            if not frames:
                return None
                
            # 准备图像
            images = []
            for frame_path in frames:
                with open(frame_path, 'rb') as f:
                    image_data = f.read()
                images.append({
                    'mime_type': 'image/jpeg',
                    'data': image_data
                })
                
            prompt = self.get_balanced_prompt(video_id)
            
            # API调用
            response = self.gemini_model.generate_content([prompt] + images)
            result = response.text
            
            # 清理临时文件
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Gemini分析失败 {video_id}: {str(e)}")
            return None
    
    def extract_key_actions(self, result_text):
        """从结果中提取key_actions"""
        try:
            # 尝试解析JSON
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            result_json = json.loads(result_text)
            return result_json.get('key_actions', '').lower()
        except:
            # 如果JSON解析失败，使用正则表达式
            key_actions_match = re.search(r'"key_actions":\s*"([^"]*)"', result_text)
            if key_actions_match:
                return key_actions_match.group(1).lower()
            return result_text.lower()
    
    def evaluate_result(self, video_id, key_actions, ground_truth_label):
        """评估结果"""
        has_ghost_probing = "ghost probing" in key_actions or "potential ghost probing" in key_actions
        ground_truth_has_ghost = ground_truth_label != "none"
        
        if has_ghost_probing and ground_truth_has_ghost:
            return "TP"  # True Positive
        elif has_ghost_probing and not ground_truth_has_ghost:
            return "FP"  # False Positive
        elif not has_ghost_probing and ground_truth_has_ghost:
            return "FN"  # False Negative
        else:
            return "TN"  # True Negative
    
    def run_analysis(self, video_limit=100):
        """运行100视频分析"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "experiment_info": {
                "timestamp": timestamp,
                "video_limit": video_limit,
                "model": "gemini-1.5-flash",
                "prompt_version": "balanced_gpt41_style",
                "output_directory": self.output_dir
            },
            "results": []
        }
        
        # 选择前N个视频进行测试
        test_videos = self.ground_truth.head(video_limit)
        
        self.logger.info(f"开始处理 {len(test_videos)} 个视频")
        
        # 统计变量
        processed_count = 0
        error_count = 0
        
        for idx, row in tqdm.tqdm(test_videos.iterrows(), total=len(test_videos)):
            video_id = row['video_id']
            ground_truth_label = row['ground_truth_label']
            
            video_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/{video_id}"
            
            if not os.path.exists(video_path):
                self.logger.warning(f"视频文件不存在: {video_path}")
                continue
                
            self.logger.info(f"处理视频 {idx+1}/{len(test_videos)}: {video_id}")
            
            # Gemini分析
            gemini_result = self.analyze_with_gemini(video_path, video_id)
            gemini_key_actions = ""
            gemini_evaluation = "ERROR"
            
            if gemini_result:
                gemini_key_actions = self.extract_key_actions(gemini_result)
                gemini_evaluation = self.evaluate_result(video_id, gemini_key_actions, ground_truth_label)
                processed_count += 1
            else:
                error_count += 1
            
            # 记录结果
            video_result = {
                "video_id": video_id,
                "ground_truth": ground_truth_label,
                "gemini_result": gemini_result,
                "key_actions": gemini_key_actions,
                "evaluation": gemini_evaluation
            }
            
            results["results"].append(video_result)
            
            # 每10个视频保存一次结果
            if (idx + 1) % 10 == 0:
                result_file = os.path.join(self.output_dir, f"gemini_100videos_results_{timestamp}.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"已保存{idx+1}个视频的中间结果")
                
        # 最终保存结果
        result_file = os.path.join(self.output_dir, f"gemini_100videos_results_{timestamp}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"处理完成: {processed_count}个成功, {error_count}个失败")
        self.generate_summary_report(results, timestamp)
        
    def generate_summary_report(self, results, timestamp):
        """生成总结报告"""
        # 计算性能指标
        stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "ERROR": 0}
        
        for result in results["results"]:
            stats[result["evaluation"]] += 1
            
        tp, fp, tn, fn = stats["TP"], stats["FP"], stats["TN"], stats["FN"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "stats": stats,
            "total_processed": tp + fp + tn + fn,
            "success_rate": (tp + fp + tn + fn) / len(results["results"]) if len(results["results"]) > 0 else 0
        }
        
        # 生成报告
        report = {
            "experiment_summary": {
                "timestamp": timestamp,
                "total_videos": len(results["results"]),
                "processed_videos": tp + fp + tn + fn,
                "model": "Gemini-1.5-flash",
                "prompt_version": "balanced_gpt41_style",
                "output_directory": self.output_dir
            },
            "performance_metrics": metrics,
            "detailed_results": results["results"]
        }
        
        # 保存报告
        report_file = os.path.join(self.output_dir, f"gemini_100videos_summary_{timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        # 生成markdown报告
        markdown_report = self.generate_markdown_report(report)
        markdown_file = os.path.join(self.output_dir, f"gemini_100videos_report_{timestamp}.md")
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
            
        # 打印总结
        print("\n" + "="*60)
        print("Gemini-1.5-flash 100视频分析总结报告")
        print("="*60)
        print(f"测试视频数量: {len(results['results'])}")
        print(f"成功处理数量: {metrics['total_processed']}")
        print(f"成功率: {metrics['success_rate']:.1%}")
        print(f"测试时间: {timestamp}")
        print("\nGemini-1.5-flash 性能 (Balanced Prompt):")
        print(f"  精确度: {metrics['precision']:.3f}")
        print(f"  召回率: {metrics['recall']:.3f}")
        print(f"  F1分数: {metrics['f1_score']:.3f}")
        print(f"  准确率: {metrics['accuracy']:.3f}")
        print(f"  统计: {metrics['stats']}")
        
        print(f"\n详细结果已保存到: {self.output_dir}")
        print(f"JSON报告: {report_file}")
        print(f"Markdown报告: {markdown_file}")
        
        self.logger.info("100视频分析完成")
        
    def generate_markdown_report(self, report):
        """生成Markdown格式报告"""
        metrics = report["performance_metrics"]
        summary = report["experiment_summary"]
        
        markdown = f"""# Gemini-1.5-flash 100视频Ghost Probing检测报告

## 实验概述

- **实验时间**: {summary['timestamp']}
- **模型**: {summary['model']}
- **Prompt版本**: {summary['prompt_version']}
- **测试视频总数**: {summary['total_videos']}
- **成功处理数**: {summary['processed_videos']}
- **成功率**: {metrics['success_rate']:.1%}

## 性能指标

| 指标 | 数值 |
|------|------|
| 精确度 (Precision) | {metrics['precision']:.3f} |
| 召回率 (Recall) | {metrics['recall']:.3f} |
| F1分数 | {metrics['f1_score']:.3f} |
| 准确率 (Accuracy) | {metrics['accuracy']:.3f} |

## 详细统计

| 分类 | 数量 |
|------|------|
| True Positives (TP) | {metrics['stats']['TP']} |
| False Positives (FP) | {metrics['stats']['FP']} |
| True Negatives (TN) | {metrics['stats']['TN']} |
| False Negatives (FN) | {metrics['stats']['FN']} |
| 错误 (ERROR) | {metrics['stats']['ERROR']} |

## 结论

本次100视频测试使用了balanced版本的GPT-4.1风格prompt，在Gemini-1.5-flash模型上取得了F1分数{metrics['f1_score']:.3f}的性能表现。

实验数据保存在: `{summary['output_directory']}`

---
*报告生成时间: {summary['timestamp']}*
"""
        return markdown

if __name__ == "__main__":
    # 设置输出目录
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gemini-1.5-flash-run1"
    
    analyzer = GeminiBalancedAnalyzer(output_dir)
    
    print(f"开始 Gemini-1.5-flash 100视频分析 (Balanced Prompt)")
    print(f"输出目录: {output_dir}")
    print("="*60)
    
    analyzer.run_analysis(video_limit=100)