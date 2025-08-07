#!/usr/bin/env python3
"""
Few-shot Ablation Study for Ghost Probing Detection
消融实验：评估few-shot learning在鬼探头检测中的作用

实验组设置：
1. Baseline组: 不使用任何few-shot examples
2. Text Few-shot组: 使用文本few-shot examples  
3. Image Few-shot组: 使用9张鬼探头图像作为visual few-shot

目标：量化评估different few-shot approaches对ghost probing detection性能的影响
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
import shutil
from pathlib import Path

# 加载环境变量
load_dotenv()

class FewshotAblationExperiment:
    def __init__(self, base_output_dir="/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation"):
        self.base_output_dir = base_output_dir
        self.setup_experiment_structure()
        self.setup_logging("main")
        self.setup_openai_api()
        self.load_ground_truth()
        self.setup_few_shot_images()
        
    def setup_experiment_structure(self):
        """创建实验目录结构"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"fewshot_ablation_{timestamp}"
        self.experiment_dir = os.path.join(self.base_output_dir, self.experiment_id)
        
        # 创建三个实验组的目录
        self.group_dirs = {
            "baseline": os.path.join(self.experiment_dir, "group1_baseline_no_fewshot"),
            "text_fewshot": os.path.join(self.experiment_dir, "group2_text_fewshot"),
            "image_fewshot": os.path.join(self.experiment_dir, "group3_image_fewshot")
        }
        
        for group_dir in self.group_dirs.values():
            os.makedirs(group_dir, exist_ok=True)
            
        # 创建对比分析目录
        self.analysis_dir = os.path.join(self.experiment_dir, "comparative_analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
    def setup_logging(self, group_name):
        """设置日志系统"""
        log_filename = os.path.join(self.experiment_dir, f"{self.experiment_id}_{group_name}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"ablation_{group_name}")
        self.logger.info(f"Few-shot Ablation Experiment 开始 - Group: {group_name}")
        
    def setup_openai_api(self):
        """设置OpenAI API配置"""
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
        
    def load_ground_truth(self):
        """加载ground truth标签"""
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
        if os.path.exists(gt_path):
            self.ground_truth = pd.read_csv(gt_path, sep='\t')
            self.logger.info(f"加载ground truth标签: {len(self.ground_truth)}个视频")
        else:
            self.logger.warning(f"Ground truth文件不存在: {gt_path}")
            self.ground_truth = None
            
    def setup_few_shot_images(self):
        """设置few-shot图像路径"""
        self.fewshot_image_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"
        
        # 验证9张图像是否存在
        expected_images = [
            "ghost_probing_sample1_before.jpg",
            "ghost_probing_sample1_during.jpg", 
            "ghost_probing_sample1_after.jpg",
            "ghost_probing_sample2_before.jpg",
            "ghost_probing_sample2_during.jpg",
            "ghost_probing_sample2_after.jpg",
            "ghost_probing_sample3_before.jpg",
            "ghost_probing_sample3_during.jpg",
            "ghost_probing_sample3_after.jpg"
        ]
        
        self.fewshot_images = []
        for img_name in expected_images:
            img_path = os.path.join(self.fewshot_image_dir, img_name)
            if os.path.exists(img_path):
                self.fewshot_images.append(img_path)
            else:
                self.logger.warning(f"Few-shot图像不存在: {img_path}")
                
        self.logger.info(f"成功加载{len(self.fewshot_images)}张few-shot图像")
        
    def get_baseline_prompt(self, video_id, frame_interval):
        """获取baseline prompt（不含few-shot examples）"""
        return f'''You are an expert in autonomous driving safety analysis, specifically focusing on "ghost probing" detection.

Ghost probing is defined as: A dangerous traffic behavior where a pedestrian, cyclist, or vehicle suddenly emerges from behind visual obstructions (like parked cars, buildings, blind spots) with minimal warning time, creating an immediate collision risk for the autonomous vehicle.

Please analyze the provided video frames from this driving video and return a comprehensive JSON analysis.

Key Analysis Requirements:
1. DETECTION TASK: Identify if ghost probing behavior occurs in this video
2. TEMPORAL ANALYSIS: When does the dangerous behavior happen (if any)
3. RISK ASSESSMENT: Evaluate the safety implications for autonomous driving
4. DETAILED DESCRIPTION: Provide comprehensive scene analysis

Your response should be a valid JSON object with the following EXACT structure:
{{
    "video_id": "{video_id}",
    "segment_id": "full_video",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "{frame_interval}.0s",
    "sentiment": "Positive/Negative/Neutral",
    "scene_theme": "Dramatic/Routine/Dangerous/Safe",
    "characters": "brief description of people in the scene with specific details",
    "summary": "comprehensive summary of the scene and what happens",
    "actions": "actions taken by the vehicle and driver responses",
    "key_objects": "numbered list of important objects and their positions",
    "key_actions": "ghost probing/left-side overtaking/right-side overtaking/none",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}

Remember: Return only a single JSON object. Be precise in ghost probing detection.'''
    
    def get_text_fewshot_prompt(self, video_id, frame_interval):
        """获取带文本few-shot examples的prompt"""
        return f'''You are an expert in autonomous driving safety analysis, specifically focusing on "ghost probing" detection.

Ghost probing is defined as: A dangerous traffic behavior where a pedestrian, cyclist, or vehicle suddenly emerges from behind visual obstructions (like parked cars, buildings, blind spots) with minimal warning time, creating an immediate collision risk for the autonomous vehicle.

Here are detailed examples to guide your analysis:

Example 1 - Pedestrian Ghost Probing:
{{
    "video_id": "example_pedestrian_ghost",
    "segment_id": "segment_000",
    "Start_Timestamp": "2.0s",
    "End_Timestamp": "8.0s",
    "sentiment": "Negative",
    "scene_theme": "Dangerous",
    "characters": "Male pedestrian in dark clothing, approximately 25 years old",
    "summary": "A male pedestrian wearing dark clothing suddenly emerges from behind a parked white truck on the right side and steps directly into the vehicle's path. The pedestrian appears from behind the obstruction with minimal warning time, creating a dangerous ghost probing situation.",
    "actions": "The self-driving vehicle immediately begins rapid deceleration in response to the unexpected pedestrian emergence from behind the parked truck.",
    "key_objects": "1) Right side: A male pedestrian, approximately 25 years old wearing dark clothing, 3 meters away, suddenly emerging from behind a parked white truck. 2) Right side: A white truck, approximately 5 meters away, creating visual obstruction.",
    "key_actions": "ghost probing",
    "next_action": {{
        "speed_control": "rapid deceleration",
        "direction_control": "keep direction",
        "lane_control": "maintain current lane"
    }}
}}

Example 2 - No Dangerous Behavior:
{{
    "video_id": "example_normal_driving",
    "segment_id": "segment_000", 
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "10.0s",
    "sentiment": "Neutral",
    "scene_theme": "Routine",
    "characters": "None visible in immediate vicinity",
    "summary": "The vehicle is driving on a clear rural road during daytime with good visibility. No pedestrians, cyclists, or vehicles creating immediate safety concerns.",
    "actions": "The self-driving vehicle maintains consistent speed and direction on the clear road with normal driving behavior.",
    "key_objects": "None requiring immediate attention",
    "key_actions": "none",
    "next_action": {{
        "speed_control": "maintain speed",
        "direction_control": "keep direction",
        "lane_control": "maintain current lane"
    }}
}}

Example 3 - Vehicle Ghost Probing:
{{
    "video_id": "example_vehicle_ghost",
    "segment_id": "segment_000",
    "Start_Timestamp": "5.0s", 
    "End_Timestamp": "12.0s",
    "sentiment": "Negative",
    "scene_theme": "Dramatic",
    "characters": "Driver of red sedan",
    "summary": "A red sedan suddenly emerges from behind a building on the left side, entering from a perpendicular side street directly into the main road. The sedan was completely hidden by the building structure until it emerged into the intersection.",
    "actions": "The self-driving vehicle immediately initiates emergency braking and slight steering adjustment to avoid collision with the suddenly appearing vehicle.",
    "key_objects": "1) Left side: A red sedan, approximately 4 meters away, suddenly emerging from behind a building. 2) Left side: A large building creating visual obstruction.",
    "key_actions": "ghost probing",
    "next_action": {{
        "speed_control": "rapid deceleration",
        "direction_control": "slight right adjustment",
        "lane_control": "maintain current lane"
    }}
}}

Now analyze the provided video and return a JSON object following the same structure:
{{
    "video_id": "{video_id}",
    "segment_id": "full_video",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "{frame_interval}.0s",
    "sentiment": "Positive/Negative/Neutral",
    "scene_theme": "Dramatic/Routine/Dangerous/Safe",
    "characters": "brief description of people in the scene with specific details",
    "summary": "comprehensive summary of the scene and what happens",
    "actions": "actions taken by the vehicle and driver responses",
    "key_objects": "numbered list of important objects and their positions",
    "key_actions": "ghost probing/left-side overtaking/right-side overtaking/none",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}

Remember: Use the examples above as guidance. Return only a single JSON object. Be precise in ghost probing detection.'''
    
    def get_image_fewshot_prompt(self, video_id, frame_interval):
        """获取带图像few-shot examples的prompt"""
        return f'''You are an expert in autonomous driving safety analysis, specifically focusing on "ghost probing" detection.

Ghost probing is defined as: A dangerous traffic behavior where a pedestrian, cyclist, or vehicle suddenly emerges from behind visual obstructions (like parked cars, buildings, blind spots) with minimal warning time, creating an immediate collision risk for the autonomous vehicle.

I have provided 9 reference images showing ghost probing examples in 3 sequences:
- Sample 1: Images 1-3 show a sequence (before, during, after) of ghost probing at intersection
- Sample 2: Images 4-6 show a sequence (before, during, after) of person emerging from building  
- Sample 3: Images 7-9 show a sequence (before, during, after) of blind spot emergence

These reference images demonstrate what ghost probing looks like in practice:
- BEFORE: Normal scene with hidden potential threat
- DURING: Critical moment when object emerges from obstruction
- AFTER: Dangerous situation with visible threat requiring immediate action

Use these visual examples to guide your analysis of the provided video frames.

Your response should be a valid JSON object with the following EXACT structure:
{{
    "video_id": "{video_id}",
    "segment_id": "full_video",
    "Start_Timestamp": "0.0s",
    "End_Timestamp": "{frame_interval}.0s",
    "sentiment": "Positive/Negative/Neutral",
    "scene_theme": "Dramatic/Routine/Dangerous/Safe",
    "characters": "brief description of people in the scene with specific details",
    "summary": "comprehensive summary of the scene and what happens",
    "actions": "actions taken by the vehicle and driver responses",
    "key_objects": "numbered list of important objects and their positions",
    "key_actions": "ghost probing/left-side overtaking/right-side overtaking/none",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}

Compare the patterns in your video with the reference images. Return only a single JSON object. Be precise in ghost probing detection based on the visual patterns shown in the examples.'''

    def save_experiment_config(self):
        """保存实验配置信息"""
        config = {
            "experiment_info": {
                "experiment_id": self.experiment_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "purpose": "Few-shot Ablation Study for Ghost Probing Detection",
                "research_questions": [
                    "Does few-shot learning improve ghost probing detection?",
                    "Which is more effective: text vs image few-shot?",
                    "How much performance gain from 9 carefully selected images?"
                ],
                "experimental_groups": {
                    "group1_baseline": {
                        "description": "No few-shot examples",
                        "prompt_type": "baseline",
                        "expected_performance": "lower baseline"
                    },
                    "group2_text_fewshot": {
                        "description": "Text-based few-shot examples",
                        "prompt_type": "text_fewshot",
                        "examples_count": 3,
                        "expected_performance": "improved over baseline"
                    },
                    "group3_image_fewshot": {
                        "description": "Visual few-shot with 9 ghost probing images",
                        "prompt_type": "image_fewshot", 
                        "examples_count": 9,
                        "image_sequences": 3,
                        "expected_performance": "best performance due to visual guidance"
                    }
                },
                "evaluation_metrics": [
                    "Precision for ghost probing detection",
                    "Recall for ghost probing detection", 
                    "F1-score for ghost probing detection",
                    "Overall accuracy",
                    "False positive rate",
                    "False negative rate"
                ],
                "model_config": {
                    "model": "GPT-4o (Azure)",
                    "temperature": 0,
                    "max_tokens": 3000,
                    "endpoint": self.vision_endpoint,
                    "deployment": self.vision_deployment
                }
            }
        }
        
        config_path = os.path.join(self.experiment_dir, "experiment_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"实验配置已保存至: {config_path}")
        return config

if __name__ == "__main__":
    experiment = FewshotAblationExperiment()
    config = experiment.save_experiment_config()
    print("Few-shot Ablation Experiment 配置完成")
    print(f"实验目录: {experiment.experiment_dir}")
    print(f"实验组: {list(experiment.group_dirs.keys())}")
    print(f"Few-shot图像: {len(experiment.fewshot_images)}张")