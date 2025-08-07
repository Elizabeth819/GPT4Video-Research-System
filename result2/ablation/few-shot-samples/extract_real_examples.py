#!/usr/bin/env python3
"""
从Run 8的结果中提取真实的高质量TP和TN样本作为few-shot examples
"""

import json
import re

def clean_json_string(json_str):
    """清理JSON字符串"""
    cleaned = json_str.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def main():
    # 读取Run 8的结果
    with open('/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8_gpt4o_ghost_probing_fewshot_100videos_results/run8_ghost_probing_100videos_results/run8_final_results_20250727_093406.json', 'r') as f:
        data = json.load(f)
    
    # 找出高质量的TP和TN样本
    tp_cases = [r for r in data['detailed_results'] if r['evaluation'] == 'TP']
    tn_cases = [r for r in data['detailed_results'] if r['evaluation'] == 'TN']
    
    print("=== TRUE POSITIVE CASES (正确识别Ghost Probing) ===")
    for i, case in enumerate(tp_cases[:3]):
        print(f"\n{i+1}. Video: {case['video_id']}")
        print(f"   Ground Truth: {case['ground_truth']}")
        
        try:
            result_json = clean_json_string(case['raw_result'])
            result = json.loads(result_json)
            print(f"   Summary: {result['summary'][:200]}...")
            print(f"   Key Actions: {result['key_actions']}")
            print(f"   Key Objects: {result['key_objects'][:150]}...")
        except Exception as e:
            print(f"   解析错误: {e}")
        print("   ---")
    
    print("\n=== TRUE NEGATIVE CASES (正确识别Normal Driving) ===")
    for i, case in enumerate(tn_cases[:3]):
        print(f"\n{i+1}. Video: {case['video_id']}")
        print(f"   Ground Truth: {case['ground_truth']}")
        
        try:
            result_json = clean_json_string(case['raw_result'])
            result = json.loads(result_json)
            print(f"   Summary: {result['summary'][:200]}...")
            print(f"   Key Actions: {result['key_actions']}")
            print(f"   Key Objects: {result['key_objects'][:150]}...")
        except Exception as e:
            print(f"   解析错误: {e}")
        print("   ---")
    
    # 统计分析
    print(f"\n=== 统计分析 ===")
    print(f"TP案例数: {len(tp_cases)}")
    print(f"TN案例数: {len(tn_cases)}")
    print(f"总样本数: {len(data['detailed_results'])}")

if __name__ == "__main__":
    main()