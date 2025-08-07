#!/usr/bin/env python3
"""
提取LLaVA检测结果，每个视频单独保存为文件
便于直观验证检测效果
"""

import json
import os
from pathlib import Path
from datetime import datetime

def extract_individual_results():
    """提取每个视频的检测结果到单独文件"""
    
    # 读取结果文件
    result_file = "./outputs/results/gpt41_balanced_100_videos_20250722_032034.json"
    
    if not Path(result_file).exists():
        print(f"❌ 结果文件不存在: {result_file}")
        return
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    metadata = data.get('metadata', {})
    
    print(f"🚀 开始提取 {len(results)} 个视频的检测结果...")
    
    # 创建输出目录
    output_dir = Path("./individual_results")
    output_dir.mkdir(exist_ok=True)
    
    # 加载ground truth用于对比
    gt_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv"
    ground_truth = {}
    
    if Path(gt_file).exists():
        import pandas as pd
        df = pd.read_csv(gt_file, sep='\t')
        for _, row in df.iterrows():
            video_id = str(row['video_id']).replace('.avi', '')
            label = str(row['ground_truth_label']).lower()
            has_ghost = (
                'ghost probing' in label or 
                'ghost' in label or
                ('s:' in label and 'none' not in label and 'cut-in' not in label)
            )
            ground_truth[video_id] = {
                'has_ghost_probing': has_ghost,
                'original_label': row['ground_truth_label']
            }
    
    # 统计信息
    ghost_detected_count = 0
    potential_detected_count = 0
    correct_detections = 0
    false_positives = 0
    missed_detections = 0
    
    # 处理每个视频
    for i, result in enumerate(results):
        video_id = result.get('video_id', f'video_{i}')
        
        # 检测分类
        key_actions = result.get('key_actions', '').lower()
        
        if 'ghost probing' in key_actions and 'potential' not in key_actions:
            detection_type = "🚨 高置信度鬼探头"
            ghost_detected_count += 1
            detected = True
        elif 'potential ghost probing' in key_actions:
            detection_type = "⚠️ 潜在鬼探头"
            potential_detected_count += 1
            detected = True
        elif 'emergency braking' in key_actions:
            detection_type = "🟡 紧急制动"
            detected = False
        else:
            detection_type = "✅ 正常交通"
            detected = False
        
        # Ground Truth对比
        gt_info = ground_truth.get(video_id, {})
        has_ground_truth_ghost = gt_info.get('has_ghost_probing', False)
        
        # 准确性判断
        if has_ground_truth_ghost and detected:
            accuracy_status = "✅ 正确检测"
            correct_detections += 1
        elif not has_ground_truth_ghost and detected:
            accuracy_status = "❌ 误报"
            false_positives += 1
        elif has_ground_truth_ghost and not detected:
            accuracy_status = "⚠️ 漏检"
            missed_detections += 1
        else:
            accuracy_status = "✅ 正确判断为正常"
        
        # 创建详细分析报告
        analysis_report = {
            "视频信息": {
                "视频ID": video_id,
                "检测类型": detection_type,
                "准确性评估": accuracy_status,
                "Ground Truth": gt_info.get('original_label', '未知'),
                "处理时间": f"{result.get('processing_time', 0):.2f}秒"
            },
            
            "检测结果详情": {
                "关键动作": result.get('key_actions', ''),
                "场景概述": result.get('summary', ''),
                "场景主题": result.get('scene_theme', ''),
                "情感色调": result.get('sentiment', ''),
                "关键对象": result.get('key_objects', ''),
                "下一步动作": result.get('next_action', {})
            },
            
            "技术分析": {
                "最大帧变化": result.get('max_frame_change', 0),
                "平均帧变化": result.get('avg_frame_change', 0),
                "帧变化序列": result.get('feature_changes', []),
                "置信度分数": result.get('gpt41_analysis', {}).get('confidence_score', 0),
                "突然变化次数": result.get('gpt41_analysis', {}).get('sudden_changes', 0),
                "分析时间": f"{result.get('analysis_time', 0):.4f}秒",
                "检测方法": result.get('gpt41_analysis', {}).get('detection_method', '')
            },
            
            "模型元数据": {
                "模型": result.get('model', ''),
                "时间戳": result.get('timestamp', ''),
                "设备": result.get('device', ''),
                "帧数": result.get('frames_analyzed', 0)
            }
        }
        
        # 保存到单独文件
        filename = f"{video_id}_analysis.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2, ensure_ascii=False)
        
        # 打印简要信息
        confidence = result.get('gpt41_analysis', {}).get('confidence_score', 0)
        max_change = result.get('max_frame_change', 0)
        
        print(f"{i+1:3d}. {video_id:15s} | {detection_type:15s} | {accuracy_status:12s} | 置信度:{confidence:.3f} | 最大变化:{max_change:.4f}")
    
    # 创建汇总报告
    summary_report = {
        "检测汇总": {
            "总视频数": len(results),
            "高置信度鬼探头": ghost_detected_count,
            "潜在鬼探头": potential_detected_count,
            "正常交通": len(results) - ghost_detected_count - potential_detected_count,
            "检测总数": ghost_detected_count + potential_detected_count
        },
        
        "准确性分析": {
            "正确检测": correct_detections,
            "误报": false_positives,
            "漏检": missed_detections,
            "Ground Truth鬼探头总数": sum(1 for gt in ground_truth.values() if gt.get('has_ghost_probing', False)),
            "检测准确率": f"{(correct_detections / len(results)) * 100:.1f}%" if results else "0%"
        },
        
        "性能指标": {
            "召回率": f"{(correct_detections / (correct_detections + missed_detections)) * 100:.1f}%" if (correct_detections + missed_detections) > 0 else "0%",
            "精确度": f"{(correct_detections / (correct_detections + false_positives)) * 100:.1f}%" if (correct_detections + false_positives) > 0 else "0%",
            "误报率": f"{(false_positives / len(results)) * 100:.1f}%" if results else "0%"
        }
    }
    
    # 保存汇总报告
    summary_file = output_dir / "detection_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*100)
    print("📊 检测结果汇总:")
    print("="*100)
    print(f"📹 总视频数: {len(results)}")
    print(f"🚨 高置信度鬼探头: {ghost_detected_count}")
    print(f"⚠️ 潜在鬼探头: {potential_detected_count}")
    print(f"✅ 正常交通: {len(results) - ghost_detected_count - potential_detected_count}")
    print(f"🎯 检测总数: {ghost_detected_count + potential_detected_count}")
    print()
    print(f"✅ 正确检测: {correct_detections}")
    print(f"❌ 误报: {false_positives}")
    print(f"⚠️ 漏检: {missed_detections}")
    print()
    print(f"📁 单独结果文件保存在: {output_dir}")
    print(f"📄 汇总报告: {summary_file}")
    print("="*100)
    
    return output_dir

if __name__ == "__main__":
    output_dir = extract_individual_results()
    print(f"\n🎉 提取完成! 请查看 {output_dir} 目录中的单个视频分析文件")