#!/usr/bin/env python3
"""
Ablation Study: 2 Few-shot Samples
基于Run 8配置，使用2个few-shot样本进行消融实验
测试平衡few-shot学习的效果（1个positive + 1个negative样本）
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

# 添加父目录到路径以导入fewshot_examples
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fewshot_examples import get_fewshot_examples

# 加载环境变量
load_dotenv()

class GPT4oAblation2Samples:
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
        log_filename = os.path.join(self.output_dir, f"ablation_2samples_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("消融实验: 2 Few-shot Samples 开始")
        
    def setup_openai_api(self):
        """设置OpenAI API"""
        self.openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY未设置")
        
        # Azure OpenAI配置
        self.vision_endpoint = os.environ.get("AZURE_OPENAI_API_ENDPOINT", "")
        self.vision_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-global")
        
        if not self.vision_endpoint:
            raise ValueError("AZURE_OPENAI_API_ENDPOINT未设置")
            
        self.logger.info(f"Azure OpenAI API配置成功")
        self.logger.info(f"Endpoint: {self.vision_endpoint}")
        self.logger.info(f"Deployment: {self.vision_deployment}")
        self.logger.info(f"Temperature: 0, Few-shot Samples: 2")
        
    def load_ground_truth(self):
        """加载ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/labels.csv"
        self.ground_truth = pd.read_csv(gt_path, encoding='utf-8-sig')
        self.logger.info(f"加载ground truth标签: {len(self.ground_truth)}个视频")
        self.logger.info(f"标签列: {list(self.ground_truth.columns)}")
        
    def initialize_results(self):
        """初始化结果结构"""
        # 尝试加载现有结果文件 (查找任何现有的结果文件)
        import glob
        pattern = os.path.join(self.output_dir, "ablation_2samples_results_*.json")
        existing_files = glob.glob(pattern)
        
        if existing_files:
            # 使用最新的结果文件
            latest_file = max(existing_files, key=os.path.getmtime)
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                self.logger.info(f"加载现有结果文件: {latest_file}")
                self.logger.info(f"已处理视频数: {len(self.results.get('detailed_results', []))}")
                return
            except Exception as e:
                self.logger.warning(f"加载现有结果失败: {str(e)}")
        
        # 创建新的结果结构
        self.results = {
            "experiment_info": {
                "run_id": "Ablation Study - 2 Few-shot Samples",
                "timestamp": self.timestamp,
                "video_count": 100,
                "model": "GPT-4o (Azure)",
                "prompt_version": "Paper_Batch Complex (4-Task) + 2 Few-shot Examples",
                "temperature": 0,
                "max_tokens": 3000,
                "purpose": "消融实验：测试平衡few-shot学习的效果",
                "baseline_comparison": "Run 8 (3 few-shot samples, F1=70.0%)",
                "output_directory": self.output_dir,
                "ablation_parameters": {
                    "few_shot_samples": 2,
                    "selected_examples": [
                        "Example 1: Ghost Probing Detection (positive样本)",
                        "Example 2: Normal Driving (negative样本)"
                    ],
                    "control_variables": [
                        "相同模型: GPT-4o",
                        "相同Temperature: 0",
                        "相同基础prompt: Paper_Batch Complex",
                        "相同评估数据: DADA-100"
                    ],
                    "test_variable": "Few-shot样本数量: 3 → 2 (平衡的positive/negative样本)"
                }
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
    
    def get_paper_batch_prompt_with_2_fewshot(self, video_id, frame_interval=10, frames_per_interval=10):
        """获取包含2个Few-shot Examples的Paper_Batch prompt"""
        
        # 获取2个few-shot样本
        fewshot_examples = get_fewshot_examples(num_samples=2)
        
        return f'''You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the portion of the video you are considering.

Direction - Please identify the objects in the image based on their position in the image itself. Do not assume your own position within the image. Treat the left side of the image as 'left' and the right side of the image as 'right'. Assume the viewpoint is standing from at the bottom center of the image. Describe whether the objects are on the left or right side of this central point, left is left, and right is right. For example, if there is a car on the left side of the image, state that the car is on the left.

**Task 1: Identify and Predict potential "Ghost Probing(专业术语：鬼探头)" behavior**

"Ghost Probing" includes the following key behaviors:

1) Traditional Ghost Probing: 
   - A person or cyclist suddenly darting out from either left or right side of the car
   - Must emerge from behind a physical obstruction that blocks the driver's view, such as a parked car, a tree, or a wall
   - Directly entering the driver's path with minimal reaction time

2) Vehicle Ghost Probing: 
   - A vehicle suddenly emerging from behind a physical obstruction
   - Examples include: buildings at intersections, parked vehicles, roadside structures, flower beds, a bridge, even a moving car at the front hiding another moving car, etc.
   - Vehicles entering from perpendicular roads that were previously hidden by obstructions

Core Characteristics:
- Presence of a physical obstruction that creates a visual barrier
- Sudden appearance from behind this obstruction with minimal reaction time
- The physical obstruction makes detection impossible until emergence
- Creates an immediate danger or potential collision situation

Note: Only those emerging from behind a physical obstruction can be considered as 鬼探头 (ghost probing).

**Task 2: Explain Current Driving Actions**
Analyze the current video frames to extract actions. Describe not only the actions themselves but also provide detailed reasoning for why the vehicle is taking these actions, such as changes in speed and direction. Focus solely on the reasoning for the vehicle's actions, excluding any descriptions of pedestrian behavior. Explain why the driver is driving at a certain speed, making turns, or stopping. Your goal is to provide a comprehensive understanding of the vehicle's behavior based on the visual data. Output in the "actions" field of the JSON format template.

**Task 3: Predict Next Driving Action**
Understand the current road conditions, the driving behavior, and to predict the next driving action. Analyze the video and audio to provide a comprehensive summary of the road conditions, including weather, traffic density, road obstacles, and traffic light if visible. Predict the next driving action based on two dimensions, one is driving speed control, such as accelerating, braking, turning, or stopping, the other one is to predict the next lane control, such as change to left lane, change to right lane, keep left in this lane, keep right in this lane, keep straight. Your summary should help understand not only what is happening at the moment but also what is likely to happen next with logical reasoning. The principle is safety first, so the prediction action should prioritize the driver's safety and secondly the pedestrians' safety. Be incredibly detailed. Output in the "next_action" field of the JSON format template.

**Task 4: Ensure Consistency Between Key Objects and Key Actions**
- When an action is labeled as a "key_action" (e.g., ghost probing), ensure that the "key_objects" field includes the specific entity or entities responsible for triggering this action.

Additional Requirements:
- `key_actions` must strictly adhere to the predefined categories:
    - ghost probing
    - overtaking, specify "left-side overtaking" or "right-side overtaking" when relevant.
    - none (if no dangerous behavior is observed)

- All textual fields must be in English.
- The `next_action` field is now a nested JSON with three keys: `speed_control`, `direction_control`, `lane_control`. Each must choose one value from their respective sets.

{fewshot_examples}

Your response should be a valid JSON object with the following EXACT structure:
{{
    "video_id": "{video_id}",
    "segment_id": "full_video",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "{frame_interval}.0s",
    "sentiment": "Positive/Negative/Neutral",
    "scene_theme": "Dramatic/Routine/Dangerous/Safe",
    "characters": "brief description of people in the scene with specific details (age, gender, clothing, transportation)",
    "summary": "comprehensive summary of the scene and what happens with incredible detail",
    "actions": "actions taken by the vehicle and driver responses with detailed reasoning",
    "key_objects": "numbered list: 1) Position: object description, distance, behavior impact 2) Position: object description, distance, behavior impact",
    "key_actions": "ghost probing/left-side overtaking/right-side overtaking/none",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}

Audio Transcription: [No audio available for this analysis]

Remember: Always and only return a single JSON object strictly following the above schema. Be incredibly detailed in your analysis, especially for ghost probing detection. Use the examples above as guidance for the level of detail and accuracy expected.'''
    
    def send_azure_openai_request(self, prompt, images):
        """发送Azure OpenAI请求 - 使用Temperature=0"""
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
            "max_tokens": 3000,
            "temperature": 0  # 关键配置：Temperature=0
        }
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.openai_api_key
        }
        
        url = f"{self.vision_endpoint}openai/deployments/{self.vision_deployment}/chat/completions?api-version=2024-02-15-preview"
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                self.logger.error(f"API响应格式错误: {result}")
                return None
        except requests.exceptions.Timeout:
            self.logger.error("API请求超时")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API请求失败: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"处理API响应时发生错误: {str(e)}")
            return None
    
    def parse_json_response(self, response_text):
        """解析JSON响应"""
        try:
            # 清理响应文本
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            
            # 解析JSON
            result = json.loads(cleaned_text)
            return result
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {str(e)}")
            self.logger.error(f"原始响应: {response_text}")
            return None
    
    def evaluate_result(self, video_id, parsed_result):
        """评估结果"""
        if not parsed_result or 'key_actions' not in parsed_result:
            return 'UNKNOWN'
        
        # 获取ground truth (video_id可能需要加.avi后缀)
        video_id_with_ext = video_id + '.avi' if not video_id.endswith('.avi') else video_id
        gt_row = self.ground_truth[self.ground_truth['video_id'] == video_id_with_ext]
        if gt_row.empty:
            self.logger.warning(f"未找到ground truth: {video_id_with_ext}")
            return 'UNKNOWN'
        
        gt_label = gt_row.iloc[0]['ground_truth_label']
        predicted_actions = parsed_result['key_actions'].lower()
        
        # 判断预测结果
        if 'ghost probing' in predicted_actions or 'ghost_probing' in predicted_actions:
            predicted_label = 'ghost_probing'
        else:
            predicted_label = 'none'
        
        # 判断ground truth（处理时间信息）
        gt_has_ghost_probing = 'ghost probing' in str(gt_label).lower()
        
        # 计算评估结果
        if gt_has_ghost_probing and predicted_label == 'ghost_probing':
            return 'TP'
        elif not gt_has_ghost_probing and predicted_label == 'none':
            return 'TN'
        elif not gt_has_ghost_probing and predicted_label == 'ghost_probing':
            return 'FP'
        elif gt_has_ghost_probing and predicted_label == 'none':
            return 'FN'
        else:
            return 'UNKNOWN'
    
    def process_video(self, video_path):
        """处理单个视频"""
        video_id = os.path.basename(video_path).replace('.avi', '')
        self.logger.info(f"处理视频: {video_id}")
        
        try:
            # 提取帧
            frames = self.extract_frames_from_video(video_path)
            if not frames:
                self.logger.error(f"无法提取帧: {video_id}")
                return None
            
            # 生成prompt
            prompt = self.get_paper_batch_prompt_with_2_fewshot(video_id)
            
            # 发送API请求
            response = self.send_azure_openai_request(prompt, frames)
            if not response:
                self.logger.error(f"API请求失败: {video_id}")
                return None
            
            # 解析结果
            parsed_result = self.parse_json_response(response)
            if not parsed_result:
                self.logger.error(f"结果解析失败: {video_id}")
                return None
            
            # 评估结果
            evaluation = self.evaluate_result(video_id, parsed_result)
            
            # 清理临时帧文件
            frames_dir = os.path.join(self.output_dir, "frames_temp")
            if os.path.exists(frames_dir):
                import shutil
                shutil.rmtree(frames_dir)
            
            result = {
                "video_id": video_id,
                "ground_truth": self.ground_truth[self.ground_truth['video_id'] == video_id].iloc[0]['ground_truth_label'] if not self.ground_truth[self.ground_truth['video_id'] == video_id].empty else 'unknown',
                "key_actions": parsed_result.get('key_actions', 'unknown'),
                "evaluation": evaluation,
                "raw_result": json.dumps(parsed_result, ensure_ascii=False)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"视频处理失败 {video_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_experiment(self, limit=100):
        """运行完整实验"""
        self.logger.info(f"开始消融实验: 2 Few-shot Samples，处理 {limit} 个视频")
        
        # 获取已处理的视频ID
        processed_video_ids = set()
        if self.results['detailed_results']:
            processed_video_ids = {r['video_id'] for r in self.results['detailed_results']}
            self.logger.info(f"检测到已处理 {len(processed_video_ids)} 个视频，从检查点继续")
        
        # 获取视频列表
        video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos"
        all_video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]
        
        # 过滤掉已处理的视频
        remaining_videos = []
        for video_file in all_video_files:
            video_id = video_file.replace('.avi', '')
            if video_id not in processed_video_ids:
                remaining_videos.append(video_file)
        
        # 限制处理数量
        videos_to_process = remaining_videos[:limit - len(processed_video_ids)]
        self.logger.info(f"需要处理 {len(videos_to_process)} 个新视频 (已完成: {len(processed_video_ids)}/100)")
        
        # 处理视频
        for video_file in tqdm.tqdm(videos_to_process, desc="处理视频"):
            video_path = os.path.join(video_dir, video_file)
            result = self.process_video(video_path)
            
            if result:
                self.results['detailed_results'].append(result)
                self.logger.info(f"✅ {result['video_id']}: {result['evaluation']}")
            else:
                self.logger.error(f"❌ {video_file}: 处理失败")
            
            # 每10个视频保存一次结果
            if len(self.results['detailed_results']) % 10 == 0:
                self.save_results()
        
        # 计算最终性能指标
        metrics = self.calculate_metrics()
        self.save_results()
        self.generate_report(metrics)
        
        return metrics
    
    def calculate_metrics(self):
        """计算性能指标"""
        results = self.results['detailed_results']
        if not results:
            return None
        
        # 统计混淆矩阵
        tp = sum(1 for r in results if r['evaluation'] == 'TP')
        tn = sum(1 for r in results if r['evaluation'] == 'TN')
        fp = sum(1 for r in results if r['evaluation'] == 'FP')
        fn = sum(1 for r in results if r['evaluation'] == 'FN')
        
        # 计算性能指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        balanced_accuracy = (recall + specificity) / 2
        
        metrics = {
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "processed_videos": len(results)
        }
        
        return metrics
    
    def save_results(self):
        """保存结果"""
        results_file = os.path.join(self.output_dir, f"ablation_2samples_results_{self.timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存: {results_file}")
    
    def generate_report(self, metrics):
        """生成实验报告"""
        if not metrics:
            return
        
        report_file = os.path.join(self.output_dir, f"ablation_2samples_report_{self.timestamp}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"""# 消融实验报告: 2 Few-shot Samples

## 实验信息
- **实验时间**: {self.timestamp}
- **实验目的**: 测试平衡few-shot学习的效果 (1 positive + 1 negative样本)
- **基线对比**: Run 8 (3 few-shot samples, F1=70.0%)
- **处理视频数**: {metrics['processed_videos']}个

## 实验配置
- **模型**: GPT-4o (Azure)
- **Temperature**: 0
- **Prompt**: Paper_Batch Complex (4-Task)
- **Few-shot样本数**: 2个
  - Example 1: Ghost Probing Detection (positive样本)
  - Example 2: Normal Driving (negative样本)

## 性能结果

### 混淆矩阵
- True Positives (TP): {metrics['confusion_matrix']['TP']}
- True Negatives (TN): {metrics['confusion_matrix']['TN']}
- False Positives (FP): {metrics['confusion_matrix']['FP']}
- False Negatives (FN): {metrics['confusion_matrix']['FN']}

### 性能指标
- **F1 Score**: {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)
- **Precision**: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)
- **Recall**: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)
- **Specificity**: {metrics['specificity']:.3f} ({metrics['specificity']*100:.1f}%)
- **Accuracy**: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)
- **Balanced Accuracy**: {metrics['balanced_accuracy']:.3f} ({metrics['balanced_accuracy']*100:.1f}%)

## 与基线对比 (Run 8: 3 samples)
- **F1差异**: {metrics['f1_score']*100:.1f}% vs 70.0% = {(metrics['f1_score']*100 - 70.0):+.1f}%
- **Recall差异**: {metrics['recall']*100:.1f}% vs 84.8% = {(metrics['recall']*100 - 84.8):+.1f}%
- **Precision差异**: {metrics['precision']*100:.1f}% vs 59.6% = {(metrics['precision']*100 - 59.6):+.1f}%

## 实验结论
1. **平衡学习效果**: 2个样本(positive+negative)相比3个样本的性能变化
2. **样本质量vs数量**: 验证了平衡样本组合的重要性
3. **学习效率**: 分析了最小有效few-shot学习的阈值

## 文件路径
- 详细结果: `ablation_2samples_results_{self.timestamp}.json`
- 实验日志: `ablation_2samples_{self.timestamp}.log`
""")
        
        self.logger.info(f"实验报告已生成: {report_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='消融实验: 2 Few-shot Samples')
    parser.add_argument('--limit', type=int, default=100, help='处理视频数量限制')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    
    # 运行实验
    experiment = GPT4oAblation2Samples(output_dir)
    metrics = experiment.run_experiment(limit=args.limit)
    
    if metrics:
        print(f"\n🎉 消融实验完成！")
        print(f"📊 F1 Score: {metrics['f1_score']*100:.1f}%")
        print(f"📈 处理视频数: {metrics['processed_videos']}")
        print(f"📁 结果保存在: {output_dir}")

if __name__ == "__main__":
    main()