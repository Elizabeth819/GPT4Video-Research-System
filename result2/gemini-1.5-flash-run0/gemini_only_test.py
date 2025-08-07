#!/usr/bin/env python3
"""
简化的Gemini测试 - 验证Gemini API功能
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

# 加载环境变量
load_dotenv()

class GeminiTester:
    def __init__(self):
        self.setup_logging()
        self.setup_gemini_api()
        self.load_ground_truth()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gemini_test_log_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Gemini测试开始")
        
    def setup_gemini_api(self):
        """设置Gemini API"""
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            self.logger.info("Gemini API配置成功")
        else:
            raise ValueError("GEMINI_API_KEY未设置")
            
    def load_ground_truth(self):
        """加载ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
        self.ground_truth = pd.read_csv(gt_path, sep='\t')
        self.logger.info(f"加载ground truth标签: {len(self.ground_truth)}个视频")
        
    def extract_frames_from_video(self, video_path, frame_interval=10, frames_per_interval=10):
        """从视频中提取帧"""
        frames_dir = "frames_temp"
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
            
    def get_gemini_prompt(self, video_id):
        """获取Gemini分析prompt"""
        return f"""You are analyzing a series of SEQUENTIAL images from a driving video to detect ghost probing events.

Ghost probing refers to sudden appearance of objects (vehicles, pedestrians, cyclists) that create immediate collision risks requiring emergency braking or swerving.

Please analyze these sequential images and determine if there are any ghost probing events.

Focus on:
1. Sudden appearance of objects from blind spots
2. Objects appearing very close (within 1-2 vehicle lengths)
3. Situations requiring immediate emergency response
4. Unpredictable movements that violate traffic expectations

Respond with a JSON object:
{{
    "video_id": "{video_id}",
    "has_ghost_probing": true/false,
    "key_actions": "description of main actions observed",
    "ghost_probing_time": "time in seconds when ghost probing occurs (if any)",
    "confidence": "high/medium/low",
    "explanation": "brief explanation of the analysis"
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
                
            prompt = self.get_gemini_prompt(video_id)
            
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
    
    def evaluate_result(self, result_text, ground_truth_label):
        """评估结果"""
        try:
            # 尝试解析JSON
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            result_json = json.loads(result_text)
            has_ghost_probing = result_json.get('has_ghost_probing', False)
            
        except:
            # 如果JSON解析失败，检查文本内容
            has_ghost_probing = "ghost probing" in result_text.lower() or "true" in result_text.lower()
            
        ground_truth_has_ghost = ground_truth_label != "none"
        
        if has_ghost_probing and ground_truth_has_ghost:
            return "TP"  # True Positive
        elif has_ghost_probing and not ground_truth_has_ghost:
            return "FP"  # False Positive
        elif not has_ghost_probing and ground_truth_has_ghost:
            return "FN"  # False Negative
        else:
            return "TN"  # True Negative
    
    def run_test(self, video_limit=5):
        """运行测试"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "test_info": {
                "timestamp": timestamp,
                "video_limit": video_limit,
                "model": "gemini-1.5-flash"
            },
            "results": []
        }
        
        # 选择前N个视频进行测试
        test_videos = self.ground_truth.head(video_limit)
        
        self.logger.info(f"开始处理 {len(test_videos)} 个视频")
        
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
            gemini_evaluation = "ERROR"
            
            if gemini_result:
                gemini_evaluation = self.evaluate_result(gemini_result, ground_truth_label)
            
            # 记录结果
            video_result = {
                "video_id": video_id,
                "ground_truth": ground_truth_label,
                "gemini_result": gemini_result,
                "evaluation": gemini_evaluation
            }
            
            results["results"].append(video_result)
            
            # 实时保存结果
            result_file = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gemini_test_results_{timestamp}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
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
            "stats": stats
        }
        
        # 生成报告
        report = {
            "test_summary": {
                "timestamp": timestamp,
                "total_videos": len(results["results"]),
                "model": "Gemini-1.5-flash"
            },
            "performance_metrics": metrics,
            "detailed_results": results["results"]
        }
        
        # 保存报告
        report_file = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gemini_test_summary_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        # 打印总结
        print("\n" + "="*50)
        print("Gemini测试总结报告")
        print("="*50)
        print(f"测试视频数量: {len(results['results'])}")
        print(f"测试时间: {timestamp}")
        print("\nGemini-1.5-flash 性能:")
        print(f"  精确度: {metrics['precision']:.3f}")
        print(f"  召回率: {metrics['recall']:.3f}")
        print(f"  F1分数: {metrics['f1_score']:.3f}")
        print(f"  准确率: {metrics['accuracy']:.3f}")
        print(f"  统计: {metrics['stats']}")
        
        print(f"\n详细结果已保存到: {report_file}")
        
        self.logger.info("测试完成")

if __name__ == "__main__":
    tester = GeminiTester()
    
    # 运行测试，首先用少量视频测试
    video_limit = 5  # 先用5个视频测试
    
    print(f"开始 Gemini-1.5-flash 测试")
    print(f"测试视频数量: {video_limit}")
    print("="*50)
    
    tester.run_test(video_limit=video_limit)