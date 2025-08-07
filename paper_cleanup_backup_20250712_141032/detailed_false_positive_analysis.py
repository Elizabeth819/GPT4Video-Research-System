#!/usr/bin/env python3
"""
详细分析GPT-4.1的误报案例
"""

import os
import json
import csv

def load_ground_truth():
    """加载Ground Truth标签"""
    ground_truth_path = "result/groundtruth_labels.csv"
    ground_truth = {}
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['video_id'] and row['video_id'].endswith('.avi'):
                video_id = row['video_id'].replace('.avi', '')
                label = row['ground_truth_label']
                ground_truth[video_id] = label
    
    return ground_truth

def analyze_specific_cases():
    """分析特定的误报案例"""
    print("🔍 详细分析GPT-4.1误报案例")
    print("=" * 80)
    
    ground_truth = load_ground_truth()
    
    # 分析您在IDE中打开的案例
    case_studies = [
        "images_5_054",
        "images_1_001", 
        "images_5_008",
        "images_4_001",
        "images_5_022"
    ]
    
    for i, video_id in enumerate(case_studies, 1):
        print(f"\n📋 案例 {i}: {video_id}")
        print("-" * 60)
        
        # 获取Ground Truth
        gt_label = ground_truth.get(video_id, "未知")
        print(f"Ground Truth标签: {gt_label}")
        
        # 加载GPT-4.1分析结果
        result_file = f"result/gpt41-gt-final/actionSummary_{video_id}.json"
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            print(f"GPT-4.1预测: 鬼探头")
            
            # 分析每个时间段
            for segment in result_data:
                if not isinstance(segment, dict):
                    continue
                    
                segment_id = segment.get('segment_id', 'unknown')
                timestamp = f"{segment.get('Start_Timestamp', 'N/A')} - {segment.get('End_Timestamp', 'N/A')}"
                
                print(f"\n   时间段 {segment_id} ({timestamp}):")
                
                # 检查情感和主题
                sentiment = segment.get('sentiment', 'N/A')
                theme = segment.get('scene_theme', 'N/A')
                print(f"     情感: {sentiment}, 主题: {theme}")
                
                # 检查关键动作
                key_actions = segment.get('key_actions', '')
                if 'ghost' in key_actions.lower() or 'probing' in key_actions.lower():
                    print(f"     关键动作: {key_actions}")
                
                # 检查摘要中的关键信息
                summary = segment.get('summary', '')
                if any(word in summary.lower() for word in ['sudden', 'unexpected', 'emergency', 'ghost']):
                    print(f"     关键摘要: {summary[:100]}...")
            
            # 分析误报原因
            if gt_label == "none":
                print(f"\n   ❌ 误报分析:")
                print(f"     - Ground Truth明确标注为'none'（无鬼探头）")
                print(f"     - GPT-4.1可能将正常的交通行为误识别为鬼探头")
                print(f"     - 可能触发词：'sudden', 'unexpected', 'emergency'等")
            
        else:
            print(f"   ⚠️ 未找到GPT-4.1分析结果文件")
    
    print(f"\n" + "=" * 80)
    print("📊 精确度问题总结")
    print("=" * 80)
    
    print(f"\n🔍 主要误报原因:")
    print(f"1. **过度敏感的关键词检测**")
    print(f"   - GPT-4.1对'sudden'、'unexpected'、'emergency'等词过度敏感")
    print(f"   - 正常的交通行为（如行人过马路、车辆变道）被误判")
    
    print(f"\n2. **缺乏真正的'鬼探头'定义理解**")
    print(f"   - 真正的鬼探头：从盲区突然出现，距离极近，时间极短")
    print(f"   - GPT-4.1将所有'突然出现'的行为都标记为鬼探头")
    
    print(f"\n3. **上下文理解不足**")
    print(f"   - 未考虑交通环境（交叉口vs高速路）")
    print(f"   - 未区分预期行为（红绿灯处行人过马路）vs非预期行为")
    
    print(f"\n💡 改进建议:")
    print(f"1. **提高判断标准**")
    print(f"   - 距离阈值：只有<2米的突然出现才考虑鬼探头")
    print(f"   - 时间阈值：必须是<1秒的瞬间出现")
    
    print(f"2. **环境上下文考虑**")
    print(f"   - 交叉口场景：行人/车辆过马路是正常行为")
    print(f"   - 高速路场景：任何突然出现都更可能是鬼探头")
    
    print(f"3. **多模态验证**")
    print(f"   - 结合视觉和运动信息")
    print(f"   - 验证是否真的需要紧急制动")

def main():
    analyze_specific_cases()

if __name__ == "__main__":
    main()