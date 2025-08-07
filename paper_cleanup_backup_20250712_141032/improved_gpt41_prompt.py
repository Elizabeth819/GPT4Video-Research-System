#!/usr/bin/env python3
"""
展示改进后的GPT-4.1 prompt，减少误报率
"""

def show_original_vs_improved_prompt():
    print("🔧 GPT-4.1 Prompt改进对比")
    print("=" * 80)
    
    print("\n📋 原始Prompt (容易误报):")
    print("-" * 60)
    
    original_prompt = """
IMPORTANT: For ghost probing detection, look for:
- Objects (people, vehicles, animals) that suddenly appear from blind spots
- Sudden movements crossing the vehicle's path
- Unexpected intrusions into the driving lane
- Objects emerging from concealed positions (behind parked cars, structures)
- Any scenario where an object "probes" or tests the vehicle's reaction

If you detect any ghost probing behavior, explicitly mention "ghost probing" in your key_actions field.
"""
    
    print(original_prompt)
    
    print("\n📋 改进后的Prompt (减少误报):")
    print("-" * 60)
    
    improved_prompt = """
IMPORTANT: For GENUINE ghost probing detection, ALL of the following criteria must be met:

1. **ENVIRONMENT CHECK**:
   - HIGH-RISK environments: Highways, rural roads, parking lots
   - LOW-RISK environments: Intersections, crosswalks, traffic lights (normal pedestrian/vehicle behavior expected)

2. **PROXIMITY & TIMING**:
   - Object appears EXTREMELY close (within 1-2 vehicle lengths)
   - Appearance is INSTANTANEOUS (not gradual approach)
   - Requires IMMEDIATE emergency braking/swerving

3. **BEHAVIOR PATTERN**:
   - Object emerges from TRUE blind spots (not visible approach paths)
   - Movement is UNPREDICTABLE and violates traffic norms
   - NOT normal traffic behaviors: pedestrians at crosswalks, vehicles changing lanes with signals, cyclists in bike lanes

4. **EXCLUSIONS** (DO NOT mark as ghost probing):
   - Pedestrians crossing at intersections/crosswalks
   - Vehicles making normal lane changes or turns
   - Cyclists following traffic patterns
   - Any scenario where the movement is EXPECTED given the environment

5. **CONFIRMATION**:
   Only mark as "ghost probing" if this creates a TRUE emergency situation that could not be reasonably anticipated.

Use "ghost probing" in key_actions ONLY when ALL above criteria are satisfied.
For other sudden but normal traffic behaviors, use terms like "emergency braking due to pedestrian crossing" or "evasive action for vehicle maneuver".
"""
    
    print(improved_prompt)
    
    print("\n" + "=" * 80)
    print("🔄 主要改进点")
    print("=" * 80)
    
    improvements = [
        "✅ 增加环境上下文判断 (交叉口 vs 高速路)",
        "✅ 提高判断门槛 (ALL criteria must be met)",
        "✅ 明确排除正常交通行为",
        "✅ 要求确认紧急程度", 
        "✅ 提供替代描述语言"
    ]
    
    for improvement in improvements:
        print(f"   {improvement}")

def create_improved_script():
    """创建使用改进prompt的脚本"""
    print(f"\n📝 创建改进版脚本文件...")
    
    script_content = '''
# 在ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper-batch-gpt41.py中
# 将第443-481行的system_content替换为:

system_content = f"""You are VideoAnalyzerGPT analyzing a series of SEQUENTIAL images taken from a video, where each image represents a consecutive moment in time. Focus on the changes in the relative positions, distances, and speeds of objects, particularly the car in front and self vehicle, and how these might indicate a potential need for braking or collision avoidance. Based on the sequence of images, predict the next action that the observer vehicle should take.

Your job is to take in as an input a transcription of {frame_interval} seconds of audio from a video,
as well as {frames_per_interval} frames split evenly throughout {frame_interval} seconds.
You are to generate and provide a Current Action Summary of the video you are considering ({frames_per_interval}
frames over {frame_interval} seconds), which is generated from your analysis of each frame ({frames_per_interval} in total),
as well as the in-between audio, until we have a full action summary of the video.

IMPORTANT: For GENUINE ghost probing detection, ALL of the following criteria must be met:

1. **ENVIRONMENT CHECK**:
   - HIGH-RISK environments: Highways, rural roads, parking lots, residential streets
   - LOW-RISK environments: Intersections, crosswalks, traffic lights (normal pedestrian/vehicle behavior expected)

2. **PROXIMITY & TIMING**:
   - Object appears EXTREMELY close (within 1-2 vehicle lengths)
   - Appearance is INSTANTANEOUS (not gradual approach)
   - Requires IMMEDIATE emergency braking/swerving

3. **BEHAVIOR PATTERN**:
   - Object emerges from TRUE blind spots (not visible approach paths)
   - Movement is UNPREDICTABLE and violates traffic norms
   - NOT normal traffic behaviors: pedestrians at crosswalks, vehicles changing lanes with signals, cyclists in bike lanes

4. **EXCLUSIONS** (DO NOT mark as ghost probing):
   - Pedestrians crossing at intersections/crosswalks
   - Vehicles making normal lane changes or turns
   - Cyclists following traffic patterns in urban areas
   - Any scenario where the movement is EXPECTED given the environment

5. **CONFIRMATION**:
   Only mark as "ghost probing" if this creates a TRUE emergency situation that could not be reasonably anticipated.

Use "ghost probing" in key_actions ONLY when ALL above criteria are satisfied.
For other sudden but normal traffic behaviors, use terms like "emergency braking due to pedestrian crossing" or "evasive action for vehicle maneuver".

Your response should be a valid JSON object with the following EXACT structure (match this format precisely):
{{
    "video_id": "{video_id}",
    "segment_id": "{segment_id_str}",
    "Start_Timestamp": "{start_time:.1f}s",
    "End_Timestamp": "{end_time:.1f}s",
    "sentiment": "Positive/Negative/Neutral",
    "scene_theme": "Dramatic/Routine/Dangerous/Safe",
    "characters": "brief description of people in the scene",
    "summary": "comprehensive summary of the scene and what happens",
    "actions": "actions taken by the vehicle and driver responses",
    "key_objects": "numbered list: 1) Position: object description, distance, behavior impact 2) Position: object description, distance, behavior impact",
    "key_actions": "brief description of most important actions (use 'ghost probing' ONLY if ALL criteria above are met)",
    "next_action": {{
        "speed_control": "rapid deceleration/deceleration/maintain speed/acceleration",
        "direction_control": "keep direction/turn left/turn right",
        "lane_control": "maintain current lane/change left/change right"
    }}
}}

Audio Transcription: {trans}
"""
'''
    
    with open("improved_gpt41_prompt_example.txt", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 改进版prompt示例已保存到: improved_gpt41_prompt_example.txt")

def main():
    show_original_vs_improved_prompt()
    create_improved_script()
    
    print(f"\n💡 实施建议:")
    print(f"1. 📝 最容易修改: 直接替换prompt文本 (5分钟)")
    print(f"2. 🧪 中等难度: 添加后处理验证逻辑 (1-2小时)")
    print(f"3. 🔬 较难修改: 精确距离/时间测量 (需要额外模型)")
    
    print(f"\n🎯 预期效果:")
    print(f"   • 误报率从47.5%降低到~20-30%")
    print(f"   • 精确度从0.53提升到~0.70-0.80")
    print(f"   • 召回率可能略有下降但仍保持高水平")

if __name__ == "__main__":
    main()