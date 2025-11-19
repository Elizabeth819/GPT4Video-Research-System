#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DriveLM风格的Graph VQA prompt处理DADA-2000视频
模拟DriveLM的Graph Visual Question Answering方法进行Ghost Probing检测
"""

import os
import json
import cv2
import base64
import requests
import subprocess
from typing import List, Dict, Any
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DriveLMStyleProcessor:
    def __init__(self):
        # 使用OpenAI API而不是Azure，更简单稳定
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = "gpt-4o"  # 使用GPT-4o进行视觉分析
        self.output_dir = "result/drivelm_style_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_drivelm_style_prompt(self) -> str:
        """
        设计DriveLM风格的Graph Visual Question Answering prompt
        基于DriveLM的方法论：Graph结构化推理 + 多步VQA
        """
        return """You are DriveLM, an advanced Graph Visual Question Answering system for autonomous driving scene understanding.

**GRAPH VQA METHODOLOGY:**
You will analyze driving scenarios through structured graph reasoning, connecting perception, prediction, planning, and behavioral elements.

**GRAPH ELEMENTS TO IDENTIFY:**
1. **EGO VEHICLE**: The autonomous vehicle's current state and trajectory
2. **TRAFFIC PARTICIPANTS**: Vehicles, pedestrians, cyclists in the scene  
3. **ROAD INFRASTRUCTURE**: Lanes, traffic signs, intersections
4. **DYNAMIC RELATIONSHIPS**: Spatial and temporal relationships between elements
5. **RISK FACTORS**: Potential collision points and safety-critical events

**MULTI-STEP REASONING PROCESS:**

**STEP 1: Scene Graph Construction**
Build a structured representation of the current driving scenario identifying all key entities and their relationships.

**STEP 2: Temporal Analysis** 
Analyze the sequence of frames to understand motion patterns and predict future states.

**STEP 3: Risk Assessment**
Evaluate potential safety-critical events, particularly focusing on sudden appearances (ghost probing).

**STEP 4: Graph-based Decision Making**
Use the constructed graph to make reasoned decisions about scene understanding.

**PRIMARY TASK: Ghost Probing Detection**
Ghost probing refers to the sudden appearance of pedestrians, vehicles, or objects that create immediate collision risk for the ego vehicle.

**DETECTION CRITERIA:**
- Sudden appearance within ego vehicle's trajectory
- Objects appearing from blind spots (behind parked cars, buildings, etc.)
- Rapid movement into the vehicle's path
- High collision risk scenarios requiring emergency response

**OUTPUT FORMAT:**
Provide your analysis in this structured format:

```json
{
    "scene_graph": {
        "ego_vehicle": "description of ego vehicle state",
        "traffic_participants": ["list of detected vehicles, pedestrians, etc."],
        "infrastructure": "road layout and traffic elements",
        "relationships": "spatial and temporal relationships"
    },
    "temporal_analysis": {
        "motion_patterns": "observed movement patterns",
        "trajectory_predictions": "predicted future states",
        "scene_evolution": "how the scene changes over time"
    },
    "risk_assessment": {
        "ghost_probing_detected": "YES/NO",
        "risk_level": "LOW/MEDIUM/HIGH/CRITICAL", 
        "risk_factors": ["list of identified risk factors"],
        "collision_probability": "assessment of collision likelihood"
    },
    "graph_reasoning": {
        "key_connections": "important graph relationships",
        "decision_logic": "reasoning process",
        "confidence_level": "HIGH/MEDIUM/LOW"
    },
    "final_decision": {
        "ghost_probing": "YES/NO",
        "explanation": "detailed reasoning for the decision",
        "recommended_action": "suggested vehicle response"
    }
}
```

**IMPORTANT NOTES:**
- Focus on sudden appearances and unexpected movements
- Consider the ego vehicle's trajectory and reaction time
- Evaluate visibility constraints and blind spots
- Prioritize safety-critical event detection
- Use graph-based reasoning to connect multiple evidence sources

Analyze the provided video frames using this Graph VQA methodology and determine if ghost probing occurs."""

    def extract_frames(self, video_path: str, interval: int = 10, max_frames: int = 10) -> List[str]:
        """提取视频关键帧"""
        logger.info(f"提取视频帧: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"视频信息: {duration:.1f}秒, {total_frames}帧, {fps:.1f}fps")
        
        frame_paths = []
        temp_dir = "frames_temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 计算采样间隔
        if duration > 0:
            time_interval = min(interval, duration / max_frames)
            frame_interval = max(1, int(fps * time_interval))
        else:
            frame_interval = max(1, total_frames // max_frames)
        
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(temp_dir, f"frame_{extracted_count:03d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                extracted_count += 1
                
            frame_count += 1
        
        cap.release()
        logger.info(f"提取了 {len(frame_paths)} 帧")
        return frame_paths

    def encode_image(self, image_path: str) -> str:
        """将图像编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def call_vision_api(self, frame_paths: List[str]) -> Dict[str, Any]:
        """调用OpenAI Vision API进行DriveLM风格分析"""
        logger.info(f"调用OpenAI Vision API分析 {len(frame_paths)} 帧")
        
        # 准备消息
        messages = [
            {
                "role": "system",
                "content": self.get_drivelm_style_prompt()
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze these sequential video frames using Graph VQA methodology to detect ghost probing events."
                    }
                ]
            }
        ]
        
        # 添加图像
        for i, frame_path in enumerate(frame_paths):
            base64_image = self.encode_image(frame_path)
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        # API调用
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0
        }
        
        url = "https://api.openai.com/v1/chat/completions"
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            logger.info("API调用成功")
            return {
                "success": True,
                "content": content,
                "usage": result.get("usage", {})
            }
            
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def parse_drivelm_response(self, response_content: str) -> Dict[str, Any]:
        """解析DriveLM风格的响应"""
        try:
            # 尝试提取JSON部分
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
            else:
                json_content = response_content
            
            parsed = json.loads(json_content)
            
            # 提取关键信息
            ghost_probing = "NO"
            confidence = "LOW"
            
            if "risk_assessment" in parsed:
                ghost_probing = parsed["risk_assessment"].get("ghost_probing_detected", "NO")
            elif "final_decision" in parsed:
                ghost_probing = parsed["final_decision"].get("ghost_probing", "NO")
                
            if "graph_reasoning" in parsed:
                confidence = parsed["graph_reasoning"].get("confidence_level", "LOW")
            
            return {
                "parsed_response": parsed,
                "ghost_probing_detected": ghost_probing,
                "confidence_level": confidence,
                "success": True
            }
            
        except Exception as e:
            logger.warning(f"JSON解析失败，使用文本分析: {e}")
            
            # 回退到文本分析
            content_lower = response_content.lower()
            
            # 检测关键词
            ghost_indicators = ["ghost probing", "sudden appearance", "collision risk", "emergency"]
            positive_indicators = ["yes", "detected", "critical", "high risk"]
            
            ghost_detected = any(indicator in content_lower for indicator in ghost_indicators)
            positive_response = any(indicator in content_lower for indicator in positive_indicators)
            
            ghost_probing = "YES" if (ghost_detected and positive_response) else "NO"
            
            return {
                "parsed_response": {"raw_content": response_content},
                "ghost_probing_detected": ghost_probing,
                "confidence_level": "MEDIUM",
                "success": False,
                "note": "Fallback text analysis used"
            }

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """处理单个视频"""
        video_name = os.path.basename(video_path).replace('.avi', '')
        logger.info(f"开始处理视频: {video_name}")
        
        result = {
            "video_id": video_name,
            "video_path": video_path,
            "timestamp": datetime.now().isoformat(),
            "method": "DriveLM_Style_Graph_VQA",
            "status": "processing"
        }
        
        try:
            # 提取帧
            frame_paths = self.extract_frames(video_path, interval=10, max_frames=10)
            
            if not frame_paths:
                result.update({
                    "status": "error",
                    "error": "No frames extracted",
                    "ghost_probing_detected": "UNKNOWN"
                })
                return result
            
            result["frames_extracted"] = len(frame_paths)
            
            # 调用API
            api_response = self.call_vision_api(frame_paths)
            
            if not api_response["success"]:
                result.update({
                    "status": "error", 
                    "error": api_response["error"],
                    "ghost_probing_detected": "UNKNOWN"
                })
                return result
            
            # 解析响应
            parsed = self.parse_drivelm_response(api_response["content"])
            
            result.update({
                "status": "completed",
                "raw_response": api_response["content"],
                "parsed_analysis": parsed["parsed_response"],
                "ghost_probing_detected": parsed["ghost_probing_detected"],
                "confidence_level": parsed["confidence_level"],
                "api_usage": api_response.get("usage", {}),
                "parsing_success": parsed["success"]
            })
            
            # 清理临时文件
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
            
            logger.info(f"完成处理: {video_name} - Ghost Probing: {parsed['ghost_probing_detected']}")
            
        except Exception as e:
            logger.error(f"处理视频时出错: {e}")
            result.update({
                "status": "error",
                "error": str(e),
                "ghost_probing_detected": "UNKNOWN"
            })
        
        return result

    def process_video_list(self, video_list: List[str], start_from: int = 0) -> List[Dict[str, Any]]:
        """批量处理视频列表"""
        logger.info(f"开始批量处理 {len(video_list)} 个视频，从第 {start_from} 个开始")
        
        results = []
        
        for i, video_path in enumerate(video_list[start_from:], start_from):
            logger.info(f"进度: {i+1}/{len(video_list)}")
            
            result = self.process_video(video_path)
            results.append(result)
            
            # 保存中间结果
            output_file = os.path.join(self.output_dir, f"drivelm_style_results_partial.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"中间结果已保存: {output_file}")
        
        return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DriveLM风格Ghost Probing检测')
    parser.add_argument('--folder', default='DADA-2000-videos', help='视频文件夹路径')
    parser.add_argument('--single', help='处理单个视频文件')
    parser.add_argument('--start-from', type=int, default=0, help='从第N个视频开始处理')
    parser.add_argument('--limit', type=int, help='限制处理视频数量')
    
    args = parser.parse_args()
    
    processor = DriveLMStyleProcessor()
    
    if args.single:
        # 处理单个视频
        if not os.path.exists(args.single):
            logger.error(f"视频文件不存在: {args.single}")
            return
            
        result = processor.process_video(args.single)
        
        output_file = os.path.join(processor.output_dir, f"single_video_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 单个视频处理完成: {output_file}")
        
    else:
        # 批量处理
        if not os.path.exists(args.folder):
            logger.error(f"视频文件夹不存在: {args.folder}")
            return
        
        # 获取视频列表
        video_files = [f for f in os.listdir(args.folder) 
                      if f.endswith('.avi') and f.startswith('images_')]
        video_files.sort()
        
        if args.limit:
            video_files = video_files[:args.limit]
        
        logger.info(f"找到 {len(video_files)} 个视频文件")
        
        # 转换为完整路径
        video_paths = [os.path.join(args.folder, f) for f in video_files]
        
        # 批量处理
        results = processor.process_video_list(video_paths, args.start_from)
        
        # 保存最终结果
        output_file = os.path.join(processor.output_dir, f"drivelm_style_final_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 统计结果
        completed = sum(1 for r in results if r['status'] == 'completed')
        ghost_detected = sum(1 for r in results 
                           if r.get('ghost_probing_detected') == 'YES')
        
        print(f"\n📊 DriveLM风格处理结果:")
        print(f"  总视频数: {len(results)}")
        print(f"  成功处理: {completed}")
        print(f"  检测到Ghost Probing: {ghost_detected}")
        print(f"  检测率: {ghost_detected/completed*100:.1f}%" if completed > 0 else "N/A")
        print(f"  结果保存在: {output_file}")

if __name__ == "__main__":
    main()