#!/usr/bin/env python3
"""
Run 11 恢复脚本 - 从中断点继续GPT-4.1 Balanced实验
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
import base64
import requests
import traceback
import sys

# 加载环境变量
load_dotenv()

class GPT41BalancedRun11Resume:
    def __init__(self, output_dir, intermediate_file):
        self.output_dir = output_dir
        self.intermediate_file = intermediate_file
        self.setup_logging()
        self.setup_openai_api()
        self.load_ground_truth()
        self.load_intermediate_results()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        log_filename = os.path.join(self.output_dir, f"run11_resume_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Run 11 恢复: GPT-4.1 + Balanced版本实验")
        
    def setup_openai_api(self):
        """设置GPT-4.1 API配置"""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY未设置")
        
        self.vision_endpoint = os.environ.get("VISION_ENDPOINT", "")
        self.vision_deployment = os.environ.get("GPT_4.1_VISION_DEPLOYMENT_NAME", "gpt-4.1")
        
        if not self.vision_endpoint:
            raise ValueError("VISION_ENDPOINT未设置")
            
        self.logger.info(f"GPT-4.1 API配置成功")
        self.logger.info(f"Endpoint: {self.vision_endpoint}")
        self.logger.info(f"Deployment: {self.vision_deployment}")
        
    def load_ground_truth(self):
        """加载ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
        self.ground_truth = pd.read_csv(gt_path, sep='\\t')
        self.logger.info(f"加载ground truth标签: {len(self.ground_truth)}个视频")
        
    def load_intermediate_results(self):
        """加载中间结果"""
        with open(self.intermediate_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
            
        processed_videos = {r['video_id'] for r in self.results['detailed_results']}
        self.logger.info(f"已处理视频数量: {len(processed_videos)}")
        self.logger.info(f"已处理视频: {sorted(list(processed_videos))}")
        
    def extract_frames_from_video(self, video_path, frame_interval=10, frames_per_interval=10):
        """从视频中提取帧"""
        frames_dir = os.path.join(self.output_dir, "frames_temp")
        os.makedirs(frames_dir, exist_ok=True)
        
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            frames = []
            timestamps = []
            
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
                    
                    frame_filename = f"{frames_dir}/frame_{interval_idx}_{frame_idx}_{frame_time:.1f}s.jpg"
                    
                    frame = clip.get_frame(frame_time)
                    cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frames.append(frame_filename)
                    timestamps.append(frame_time)
            
            clip.close()
            return frames, timestamps
            
        except Exception as e:
            self.logger.error(f"帧提取失败 {video_path}: {str(e)}")
            return [], []
    
    def get_gpt41_balanced_prompt(self, video_id, start_time=0, end_time=10, frame_interval=10, frames_per_interval=10):
        """获取GPT-4.1 Balanced prompt"""
        segment_id = "full_video"
        
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
        """发送Azure OpenAI请求"""
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
        
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 2000,
            "temperature": 0
        }
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.openai_api_key
        }
        
        url = f"{self.vision_endpoint}openai/deployments/{self.vision_deployment}/chat/completions?api-version=2024-02-15-preview"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=90)  # 增加超时时间
                response.raise_for_status()
                
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    self.logger.error(f"API响应格式错误: {result}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        
        return None
    
    def analyze_with_gpt41_balanced(self, video_path, video_id):
        """使用GPT-4.1 Balanced进行视频分析"""
        try:
            frames, timestamps = self.extract_frames_from_video(video_path)
            if not frames:
                return None
                
            prompt = self.get_gpt41_balanced_prompt(video_id)
            
            result = self.send_azure_openai_request(prompt, frames)
            
            # 清理临时文件
            for frame_path in frames:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"视频分析失败 {video_id}: {str(e)}")
            return None
    
    def extract_video_id(self, video_path):
        """从视频路径提取video_id"""
        filename = os.path.basename(video_path)
        if filename.startswith('images_') and filename.endswith('.avi'):
            return filename[:-4]
        return filename
    
    def calculate_performance_metrics(self, results):
        """计算性能指标"""
        tp = fp = tn = fn = errors = 0
        
        for result in results:
            if result['status'] == 'error' or result['status'] == 'parse_error':
                errors += 1
                continue
                
            predicted = result['predicted_label']
            actual = result['actual_label']
            
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
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'errors': errors,
            'precision': precision, 'recall': recall, 'f1_score': f1, 'accuracy': accuracy
        }
    
    def resume_experiment(self):
        """恢复实验"""
        self.logger.info("恢复Run 11: GPT-4.1+Balanced实验")
        
        # 获取已处理的视频ID列表
        processed_videos = {r['video_id'] for r in self.results['detailed_results']}
        
        # 获取所有视频列表
        video_base_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
        all_video_files = []
        
        for i in range(1, 6):
            for j in range(1, 100):
                video_name = f"images_{i}_{j:03d}.avi"
                video_path = os.path.join(video_base_dir, video_name)
                if os.path.exists(video_path):
                    all_video_files.append(video_path)
                if len(all_video_files) >= 100:
                    break
            if len(all_video_files) >= 100:
                break
        
        # 找到未处理的视频
        remaining_videos = []
        for video_path in all_video_files:
            video_id = self.extract_video_id(video_path)
            if video_id not in processed_videos:
                remaining_videos.append(video_path)
        
        self.logger.info(f"剩余未处理视频数量: {len(remaining_videos)}")
        
        if not remaining_videos:
            self.logger.info("所有视频已处理完成")
            self.save_final_results()
            return self.results
        
        progress_bar = tqdm.tqdm(total=len(remaining_videos), desc="继续处理视频", unit="video")
        
        for i, video_path in enumerate(remaining_videos):
            video_id = self.extract_video_id(video_path)
            self.logger.info(f"处理剩余视频 {i+1}/{len(remaining_videos)}: {video_id}")
            
            start_time = time.time()
            
            # 获取ground truth标签
            gt_row = self.ground_truth[self.ground_truth['video_id'] == video_id]
            actual_label = 1 if len(gt_row) > 0 and gt_row.iloc[0]['has_ghost_probing'] else 0
            
            # 分析视频
            result = self.analyze_with_gpt41_balanced(video_path, video_id)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result is None:
                self.logger.error(f"视频 {video_id} 分析失败")
                self.results["detailed_results"].append({
                    "video_id": video_id,
                    "status": "error",
                    "actual_label": actual_label,
                    "predicted_label": 0,
                    "processing_time": processing_time,
                    "error": "API调用失败"
                })
            else:
                try:
                    parsed_result = json.loads(result)
                    
                    key_actions = parsed_result.get('key_actions', '').lower()
                    predicted_label = 1 if 'ghost probing' in key_actions else 0
                    
                    self.results["detailed_results"].append({
                        "video_id": video_id,
                        "status": "success",
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "processing_time": processing_time,
                        "raw_response": result,
                        "parsed_result": parsed_result,
                        "key_actions": key_actions
                    })
                    
                    self.logger.info(f"视频 {video_id} 完成 - 预测: {predicted_label}, 实际: {actual_label}, 用时: {processing_time:.1f}s")
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"视频 {video_id} JSON解析失败: {str(e)}")
                    self.results["detailed_results"].append({
                        "video_id": video_id,
                        "status": "parse_error",
                        "actual_label": actual_label,
                        "predicted_label": 0,
                        "processing_time": processing_time,
                        "raw_response": result,
                        "error": f"JSON解析失败: {str(e)}"
                    })
            
            progress_bar.update(1)
            
            # 每5个视频保存一次
            if (i + 1) % 5 == 0:
                self.save_intermediate_results()
        
        progress_bar.close()
        
        # 计算最终性能指标
        metrics = self.calculate_performance_metrics(self.results["detailed_results"])
        self.results["performance_metrics"] = metrics
        
        self.save_final_results()
        
        self.logger.info("Run 11实验完成")
        self.logger.info(f"性能指标: F1={metrics['f1_score']:.3f}, 召回率={metrics['recall']:.3f}, 精确度={metrics['precision']:.3f}")
        
        return self.results
    
    def save_intermediate_results(self):
        """保存中间结果"""
        filename = os.path.join(self.output_dir, f"run11_intermediate_updated_{self.timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
    
    def save_final_results(self):
        """保存最终结果"""
        json_filename = os.path.join(self.output_dir, f"run11_final_results_{self.timestamp}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        metrics = self.results["performance_metrics"]
        report_filename = os.path.join(self.output_dir, f"run11_performance_report_{self.timestamp}.md")
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"""# Run 11: GPT-4.1+Balanced 100视频复现实验 - 最终报告

## 实验信息
- **运行时间**: {self.timestamp}
- **模型**: GPT-4.1 (via Azure)
- **Prompt版本**: Balanced (Historical Best Recreation)
- **温度**: 0
- **视频数量**: {len(self.results["detailed_results"])}

## 性能指标
- **F1分数**: {metrics['f1_score']:.3f}
- **精确度**: {metrics['precision']:.3f} 
- **召回率**: {metrics['recall']:.3f}
- **准确率**: {metrics['accuracy']:.3f}

## 混淆矩阵
- **TP (True Positive)**: {metrics['tp']}
- **FP (False Positive)**: {metrics['fp']}
- **TN (True Negative)**: {metrics['tn']}
- **FN (False Negative)**: {metrics['fn']}
- **ERROR**: {metrics['errors']}

## 与历史最佳对比
### 目标指标 (GPT-4.1+Balanced历史最佳)
- F1分数: 0.712
- 召回率: 0.963 
- 精确度: 0.565

### 复现结果
- F1分数: {metrics['f1_score']:.3f} ({'+' if metrics['f1_score'] >= 0.712 else ''}{metrics['f1_score'] - 0.712:+.3f})
- 召回率: {metrics['recall']:.3f} ({'+' if metrics['recall'] >= 0.963 else ''}{metrics['recall'] - 0.963:+.3f})
- 精确度: {metrics['precision']:.3f} ({'+' if metrics['precision'] >= 0.565 else ''}{metrics['precision'] - 0.565:+.3f})

## 复现成功度评估
- F1分数: {'✅ 成功复现' if metrics['f1_score'] >= 0.712 * 0.95 else '❌ 未达到目标'}
- 召回率: {'✅ 成功复现' if metrics['recall'] >= 0.963 * 0.95 else '❌ 未达到目标'}
- 精确度: {'✅ 成功复现' if metrics['precision'] >= 0.565 * 0.95 else '❌ 未达到目标'}

## 总体结论
{'✅ 成功复现历史最佳GPT-4.1+Balanced性能' if metrics['f1_score'] >= 0.712 * 0.95 and metrics['recall'] >= 0.963 * 0.95 else '❌ 未能完全复现历史最佳性能，但提供了有价值的对比数据'}

## 技术洞察
1. **Prompt效果**: GPT-4.1+Balanced prompt在当前Azure环境下的表现
2. **API稳定性**: Azure GPT-4.1 API的响应时间和错误率
3. **复现可行性**: 历史最佳结果的可重现性分析
""")
        
        self.logger.info(f"最终结果已保存到: {json_filename}")
        self.logger.info(f"报告已保存到: {report_filename}")

def main():
    """主函数"""
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run11-gpt41-balanced-100videos"
    intermediate_file = os.path.join(output_dir, "run11_intermediate_20250727_153902.json")
    
    if not os.path.exists(intermediate_file):
        print(f"中间结果文件不存在: {intermediate_file}")
        return
    
    experiment = GPT41BalancedRun11Resume(output_dir, intermediate_file)
    results = experiment.resume_experiment()
    
    return results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n实验被用户中断")
    except Exception as e:
        print(f"实验失败: {str(e)}")
        traceback.print_exc()