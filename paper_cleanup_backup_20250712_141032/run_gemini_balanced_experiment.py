#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速启动Gemini平衡版Prompt实验
基于现有的ActionSummary-gemini.py，替换prompt为GPT-4.1平衡版
"""

import os
import sys
import json
import subprocess
from datetime import datetime

def modify_gemini_prompt():
    """
    修改ActionSummary-gemini.py中的prompt为平衡版
    """
    
    # 读取原始文件
    with open('ActionSummary-gemini.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # GPT-4.1平衡版prompt
    balanced_prompt = '''You are an expert AI system analyzing sequential video frames from autonomous driving scenarios. Your primary task is to detect "ghost probing" events using a balanced layered detection strategy.

**DEFINITION: Ghost Probing**
A dangerous traffic scenario where pedestrians, cyclists, or objects suddenly appear from concealed positions (behind parked cars, walls, blind spots) creating immediate collision risk requiring emergency braking or avoidance.

**LAYERED DETECTION STRATEGY:**

**1. HIGH-CONFIDENCE Ghost Probing (use "ghost probing" in key_actions)**:
- Object appears EXTREMELY close (within 1-2 vehicle lengths, <3 meters) 
- Appearance is SUDDEN and from blind spots (behind parked cars, walls, obstacles)
- Occurs in HIGH-RISK environments: highways, rural roads, parking lots
- Creates IMMEDIATE danger requiring emergency response
- Object was previously completely hidden and suddenly emerges

**2. POTENTIAL Ghost Probing (use "potential ghost probing" in key_actions)**:
- Object appears suddenly but at moderate distance (3-5 meters)
- Sudden movement in environments where some unpredictability exists
- Appears from partially concealed positions
- Creates heightened caution but not immediate emergency

**3. Normal Traffic (use "none" in key_actions)**:
- Predictable pedestrian crossings at crosswalks
- Cyclists in designated bike lanes
- Normal traffic flow and lane changes
- Expected movements in urban environments

**ANALYSIS FRAMEWORK:**
1. **Concealment Assessment**: Was the object previously hidden behind obstacles?
2. **Distance Evaluation**: How close is the object when first detected?
3. **Environment Context**: Is this a high-risk scenario location?
4. **Predictability**: Was this movement expected or sudden?
5. **Emergency Level**: Does this require immediate evasive action?

Your job is to analyze {frames_per_interval} frames spanning {frame_interval} seconds and provide detailed analysis.

**TASKS:**
1. **Ghost Probing Detection**: Apply the layered detection strategy
2. **Current Action Analysis**: Describe what's happening in the video
3. **Next Action Prediction**: Predict required vehicle response
4. **Object-Action Consistency**: Ensure key_objects match key_actions

**Task 2: Explain Current Driving Actions**
Analyze the current actions in the video frames, detailing why the vehicle is moving at a certain speed or direction.

**Task 3: Predict Next Driving Action**
Based on what you see, predict the most likely next actions in terms of speed control and lane control.

**Task 4: Ensure Consistency Between Key Objects and Key Actions**
When labeling a key action (like ghost probing), make sure to include the relevant objects causing this action.

Always return a single JSON object with the following fields:
- video_id: "{video_id}"
- segment_id: "{segment_id_str}"
- Start_Timestamp and End_Timestamp: derived from frame names
- summary: detailed description of what's happening
- actions: explanation of current vehicle actions
- key_objects: list of important objects affecting the vehicle
- key_actions: danger classification using layered strategy ("ghost probing", "potential ghost probing", or "none")
- next_action: JSON object with speed_control, direction_control, and lane_control fields

**IMPORTANT**: Use the layered detection strategy to maintain high recall (detect real dangers) while improving precision (reduce false positives). When in doubt between categories, prefer the more conservative classification.

All text must be in English. Return only valid JSON.'''
    
    # 找到并替换原始prompt
    # 查找system_prompt的开始和结束
    start_marker = 'system_prompt = f"""You are VideoAnalyzerGPT'
    end_marker = 'All text must be in English. Return only valid JSON."""'
    
    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker, start_pos) + len(end_marker)
    
    if start_pos == -1 or end_pos == -1:
        print("❌ 无法找到原始prompt位置")
        return False
    
    # 构建新的prompt部分
    new_prompt_section = f'system_prompt = f"""{balanced_prompt}"""'
    
    # 替换内容
    new_content = content[:start_pos] + new_prompt_section + content[end_pos:]
    
    # 保存修改后的文件
    backup_filename = f'ActionSummary-gemini-backup-{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
    
    # 备份原文件
    with open(backup_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 原文件已备份为: {backup_filename}")
    
    # 写入修改后的文件
    with open('ActionSummary-gemini-balanced-temp.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("✅ 已创建平衡版Gemini脚本: ActionSummary-gemini-balanced-temp.py")
    
    return True

def run_gemini_experiment(limit=5):
    """
    运行Gemini平衡版实验
    """
    print(f"🚀 启动Gemini平衡版Prompt实验 (处理{limit}个视频)")
    
    # 创建输出目录
    output_dir = "result/gemini-balanced-prompt"
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行实验
    cmd = [
        sys.executable, 
        "ActionSummary-gemini-balanced-temp.py",
        "DADA-2000-videos",
        "10",  # interval
        "10",  # frames
        "False",  # speed_mode
        "--output_dir", output_dir,
        "--limit", str(limit)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        if result.returncode == 0:
            print("✅ Gemini实验执行成功!")
            print("📊 输出:")
            print(result.stdout)
        else:
            print("❌ Gemini实验执行失败!")
            print("错误信息:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏱️ 实验超时 (1小时)")
        return False
    except Exception as e:
        print(f"❌ 执行过程中出错: {str(e)}")
        return False

def analyze_results():
    """
    分析实验结果
    """
    output_dir = "result/gemini-balanced-prompt"
    
    if not os.path.exists(output_dir):
        print("❌ 结果目录不存在")
        return
    
    # 统计处理的视频数量
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    print(f"\n📊 实验结果分析:")
    print(f"  处理的视频数量: {len(json_files)}")
    
    if json_files:
        print(f"  输出目录: {output_dir}")
        print(f"  示例文件: {json_files[0]}")
        
        # 读取一个示例文件
        try:
            with open(os.path.join(output_dir, json_files[0]), 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            
            print(f"  示例数据结构:")
            if isinstance(sample_data, list) and len(sample_data) > 0:
                sample_segment = sample_data[0]
                print(f"    video_id: {sample_segment.get('video_id', 'N/A')}")
                print(f"    key_actions: {sample_segment.get('key_actions', 'N/A')}")
                print(f"    summary: {sample_segment.get('summary', 'N/A')[:100]}...")
        except Exception as e:
            print(f"  ⚠️ 无法解析示例文件: {str(e)}")

def main():
    print("🎯 Gemini 2.0 Flash + GPT-4.1平衡版Prompt 公平对比实验")
    print("=" * 60)
    
    # 步骤1: 修改prompt
    print("\n步骤1: 准备平衡版prompt...")
    if not modify_gemini_prompt():
        print("❌ prompt修改失败")
        return
    
    # 步骤2: 运行实验 
    print("\n步骤2: 运行实验...")
    success = run_gemini_experiment(limit=3)  # 先测试3个视频
    
    if not success:
        print("❌ 实验失败")
        return
    
    # 步骤3: 分析结果
    print("\n步骤3: 分析结果...")
    analyze_results()
    
    print("\n🎉 实验完成!")
    print("📋 下一步:")
    print("  1. 检查结果质量")
    print("  2. 如果满意，运行完整的99个视频实验")
    print("  3. 进行公平对比分析")

if __name__ == "__main__":
    main()