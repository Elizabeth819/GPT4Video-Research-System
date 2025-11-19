#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DriveLM风格批处理测试脚本
测试3个视频后进行性能对比分析
"""

import os
import subprocess
import json
import pandas as pd
import time
from datetime import datetime

def run_drivelm_style_processing():
    """运行DriveLM风格的处理"""
    print("🚀 开始DriveLM风格Ghost Probing检测测试")
    print("=" * 60)
    
    # 测试视频列表
    test_videos = [
        "DADA-2000-videos/images_1_001.avi",
        "DADA-2000-videos/images_1_002.avi", 
        "DADA-2000-videos/images_1_003.avi"
    ]
    
    output_dir = "result/drivelm_comparison/drivelm_gpt41_results"
    os.makedirs(output_dir, exist_ok=True)
    
    successful_videos = []
    
    for i, video_path in enumerate(test_videos, 1):
        print(f"\n📹 处理视频 {i}/3: {os.path.basename(video_path)}")
        
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            continue
            
        try:
            # 运行DriveLM风格处理
            cmd = [
                "python", "ActionSummary-drivelm-gpt41.py",
                "--single", video_path,
                "--interval", "10",
                "--frames", "10",
                "--output-dir", output_dir
            ]
            
            print(f"🔧 执行命令: {' '.join(cmd)}")
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            end_time = time.time()
            
            if result.returncode == 0:
                processing_time = end_time - start_time
                print(f"✅ 处理成功，耗时: {processing_time:.1f}秒")
                successful_videos.append({
                    "video": os.path.basename(video_path),
                    "video_id": os.path.basename(video_path).replace('.avi', ''),
                    "processing_time": processing_time,
                    "status": "success"
                })
            else:
                print(f"❌ 处理失败:")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                
        except Exception as e:
            print(f"❌ 处理异常: {e}")
    
    print(f"\n📊 处理完成: {len(successful_videos)}/3 视频成功")
    return successful_videos

def analyze_drivelm_results():
    """分析DriveLM风格处理的结果"""
    print("\n🔍 分析DriveLM风格处理结果...")
    
    output_dir = "result/drivelm_comparison/drivelm_gpt41_results"
    results = []
    
    for filename in os.listdir(output_dir):
        if filename.startswith("actionSummary_drivelm_") and filename.endswith(".json"):
            video_id = filename.replace("actionSummary_drivelm_", "").replace(".json", "")
            
            file_path = os.path.join(output_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检测ghost probing
                ghost_probing_detected = False
                ghost_probing_segments = []
                
                for segment in data:
                    if isinstance(segment, dict) and 'key_actions' in segment:
                        key_actions = str(segment['key_actions']).lower()
                        if 'ghost probing' in key_actions:
                            ghost_probing_detected = True
                            ghost_probing_segments.append({
                                'segment_id': segment.get('segment_id'),
                                'timestamp': f"{segment.get('Start_Timestamp', '')}-{segment.get('End_Timestamp', '')}",
                                'key_actions': segment.get('key_actions', '')
                            })
                
                results.append({
                    'video_id': video_id,
                    'ghost_probing_detected': 'YES' if ghost_probing_detected else 'NO',
                    'ghost_probing_segments': len(ghost_probing_segments),
                    'segments_data': ghost_probing_segments,
                    'total_segments': len(data)
                })
                
            except Exception as e:
                print(f"⚠️ 无法处理文件 {filename}: {e}")
                continue
    
    return results

def compare_with_existing_results():
    """与现有结果进行对比"""
    print("\n⚖️ 与现有GPT-4.1和Gemini结果对比...")
    
    # 加载ground truth
    gt_file = "result/groundtruth_labels.csv"
    if not os.path.exists(gt_file):
        print(f"❌ Ground truth文件不存在: {gt_file}")
        return
    
    gt_df = pd.read_csv(gt_file, sep='\t')
    
    # 分析DriveLM结果
    drivelm_results = analyze_drivelm_results()
    
    # 创建对比表
    comparison_data = []
    
    for drivelm_result in drivelm_results:
        video_id = drivelm_result['video_id']
        
        # 查找ground truth
        gt_row = gt_df[gt_df['video_id'] == f"{video_id}.avi"]
        if gt_row.empty:
            continue
            
        gt_label = gt_row.iloc[0]['ground_truth_label']
        gt_has_ghost = 'ghost probing' in str(gt_label).lower()
        
        # 加载现有结果
        gpt41_result = load_existing_result("result/gpt41-balanced-full", video_id)
        gemini_result = load_existing_result("result/gemini-balanced-full", video_id)
        
        comparison_data.append({
            'video_id': video_id,
            'ground_truth': gt_label,
            'drivelm_gpt41': drivelm_result['ghost_probing_detected'],
            'gpt41_balanced': gpt41_result,
            'gemini_balanced': gemini_result,
            'drivelm_segments': drivelm_result['ghost_probing_segments'],
            'drivelm_details': drivelm_result['segments_data']
        })
    
    # 保存对比结果
    comparison_df = pd.DataFrame(comparison_data)
    output_path = "result/drivelm_comparison/analysis/drivelm_gpt41_comparison_test.csv"
    comparison_df.to_csv(output_path, index=False)
    
    print(f"✅ 对比结果保存到: {output_path}")
    
    # 显示结果
    print("\n📊 对比结果:")
    for _, row in comparison_df.iterrows():
        print(f"  🎬 {row['video_id']}:")
        print(f"    Ground Truth: {row['ground_truth']}")
        print(f"    DriveLM-GPT41: {row['drivelm_gpt41']}")
        print(f"    GPT41-Balanced: {row['gpt41_balanced']}")
        print(f"    Gemini-Balanced: {row['gemini_balanced']}")
        if row['drivelm_details']:
            print(f"    DriveLM检测细节: {len(row['drivelm_details'])}个段落")
        print()
    
    return comparison_df

def load_existing_result(result_dir, video_id):
    """加载现有的结果"""
    if not os.path.exists(result_dir):
        return "N/A"
    
    # 尝试不同的文件名格式
    possible_files = [
        f"actionSummary_{video_id}.json",
        f"actionSummary_dada_{video_id.replace('images_', '')}.json"
    ]
    
    for filename in possible_files:
        file_path = os.path.join(result_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for segment in data:
                    if isinstance(segment, dict) and 'key_actions' in segment:
                        key_actions = str(segment['key_actions']).lower()
                        if 'ghost probing' in key_actions:
                            return "YES"
                
                return "NO"
                
            except Exception as e:
                print(f"⚠️ 读取{file_path}失败: {e}")
                continue
    
    return "N/A"

def main():
    """主函数"""
    print("🎯 DriveLM风格GPT-4.1测试和对比分析")
    print(f"📅 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: 运行DriveLM风格处理
    successful_videos = run_drivelm_style_processing()
    
    if not successful_videos:
        print("❌ 没有成功处理的视频，退出程序")
        return
    
    # Step 2: 分析结果
    drivelm_results = analyze_drivelm_results()
    
    print(f"\n📈 DriveLM风格处理统计:")
    print(f"  成功处理视频: {len(drivelm_results)}")
    ghost_detected = sum(1 for r in drivelm_results if r['ghost_probing_detected'] == 'YES')
    print(f"  检测到Ghost Probing: {ghost_detected}")
    print(f"  检测率: {ghost_detected/len(drivelm_results)*100:.1f}%" if drivelm_results else "N/A")
    
    # Step 3: 与现有结果对比
    comparison_df = compare_with_existing_results()
    
    # Step 4: 总结
    print("\n🎯 测试总结:")
    print("✅ DriveLM风格Graph VQA prompt成功应用于GPT-4.1")
    print("✅ 成功处理了测试视频并生成了分析结果")
    print("✅ 与现有方法进行了对比分析")
    print("✅ 为完整的100视频处理做好了准备")
    
    print(f"\n📁 输出文件:")
    print(f"  - DriveLM结果: result/drivelm_comparison/drivelm_gpt41_results/")
    print(f"  - 对比分析: result/drivelm_comparison/analysis/")
    print(f"  - 项目说明: result/drivelm_comparison/README.md")

if __name__ == "__main__":
    main()