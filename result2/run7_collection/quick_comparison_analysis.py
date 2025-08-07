#!/usr/bin/env python3
"""
快速对比分析：基于已有Run 7结果分析前20个视频的表现
对比重新实现版本 vs 可能的原始脚本表现
"""

import json
import pandas as pd
from collections import Counter
import os

def load_run7_results():
    """加载Run 7的结果"""
    run7_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run7-gpt4o-paper-batch-temp0"
    
    # 找到最新的中间结果文件
    files = [f for f in os.listdir(run7_dir) if f.startswith("run7_intermediate_")]
    if not files:
        return None
        
    latest_file = sorted(files)[-1]
    file_path = os.path.join(run7_dir, latest_file)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_ground_truth():
    """加载ground truth"""
    gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
    return pd.read_csv(gt_path, sep='\t')

def analyze_first_20_videos():
    """分析前20个视频的表现"""
    run7_results = load_run7_results()
    ground_truth = load_ground_truth()
    
    if not run7_results:
        print("❌ 无法加载Run 7结果")
        return
    
    # 获取前20个视频的结果
    first_20_videos = [
        "images_1_001.avi", "images_1_002.avi", "images_1_003.avi", "images_1_004.avi", "images_1_005.avi",
        "images_1_006.avi", "images_1_007.avi", "images_1_008.avi", "images_1_009.avi", "images_1_010.avi",
        "images_1_011.avi", "images_1_012.avi", "images_1_013.avi", "images_1_014.avi", "images_1_015.avi",
        "images_1_016.avi", "images_1_017.avi", "images_1_018.avi", "images_1_019.avi", "images_1_020.avi"
    ]
    
    # 从Run 7结果中提取前20个视频的数据
    first_20_results = []
    for result in run7_results["detailed_results"]:
        if result["video_id"] in first_20_videos:
            first_20_results.append(result)
    
    # 按照顺序排序
    first_20_results.sort(key=lambda x: first_20_videos.index(x["video_id"]))
    
    print(f"🔍 分析前20个视频的Run 7表现")
    print("=" * 60)
    print(f"找到 {len(first_20_results)} 个视频的结果")
    
    # 计算性能指标
    evals = [r['evaluation'] for r in first_20_results]
    eval_counts = Counter(evals)
    
    tp = eval_counts.get('TP', 0)
    fp = eval_counts.get('FP', 0)
    tn = eval_counts.get('TN', 0)
    fn = eval_counts.get('FN', 0)
    errors = eval_counts.get('ERROR', 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print(f"\n📊 重新实现版本 (Run 7前20个视频) 性能:")
    print(f"   精确度: {precision:.3f} ({precision*100:.1f}%)")
    print(f"   召回率: {recall:.3f} ({recall*100:.1f}%)")
    print(f"   F1分数: {f1:.3f} ({f1*100:.1f}%)")
    print(f"   准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, ERROR: {errors}")
    
    # 详细分析每个视频
    print(f"\n📋 详细分析:")
    print("视频ID\t\t\tGround Truth\t\t检测结果\t\t评估")
    print("-" * 80)
    
    for result in first_20_results:
        video_id = result["video_id"]
        ground_truth = result["ground_truth"]
        key_actions = result["key_actions"]
        evaluation = result["evaluation"]
        
        # 截断显示
        gt_display = ground_truth[:15] + "..." if len(ground_truth) > 15 else ground_truth
        ka_display = key_actions[:15] + "..." if len(key_actions) > 15 else key_actions
        
        print(f"{video_id}\t{gt_display:<15}\t{ka_display:<15}\t{evaluation}")
    
    # 与历史结果对比
    print(f"\n🔄 与其他实验对比:")
    print("| 实验版本 | 视频数 | F1分数 | 精确度 | 召回率 | 备注 |")
    print("|----------|--------|--------|--------|--------|------|")
    print(f"| Run 7前20视频 | {len(first_20_results)} | {f1:.3f} | {precision:.3f} | {recall:.3f} | 重新实现+Temperature=0 |")
    print("| Run 5 Early | 10 | 0.800 | 0.750 | 0.857 | 原始复杂prompt |")
    print("| Run 5 Balanced | 10 | 0.600 | 1.000 | 0.429 | 简化平衡prompt |")
    print("| Run 6 完整 | 100 | 0.636 | 0.554 | 0.745 | Temperature=0.3 |")
    
    # 分析趋势
    print(f"\n🔎 关键观察:")
    print(f"1. **F1分数表现**: {f1:.3f} - {'优秀' if f1 > 0.7 else '良好' if f1 > 0.6 else '需要改进'}")
    print(f"2. **精确度**: {precision:.3f} - {'高精确度，误报少' if precision > 0.7 else '中等精确度' if precision > 0.5 else '精确度较低'}")
    print(f"3. **召回率**: {recall:.3f} - {'高召回率，漏检少' if recall > 0.8 else '中等召回率' if recall > 0.6 else '召回率较低，有安全风险'}")
    
    # 对比原始脚本可能的表现
    print(f"\n🤔 原始脚本 vs 重新实现版本分析:")
    print("**原始脚本的优势 (理论上):**")
    print("- ✅ 经过长期验证和调优")  
    print("- ✅ 完整的Paper_Batch 4任务prompt")
    print("- ✅ 可能包含特殊的边界情况处理")
    print("- ✅ 更稳定的API调用机制")
    
    print("\n**重新实现版本的问题 (可能):**")
    print("- ⚠️ Prompt可能有细微差异")
    print("- ⚠️ API调用参数可能不完全一致")
    print("- ⚠️ 图像处理流程可能简化")
    print("- ⚠️ 错误处理机制可能不够完善")
    
    # 建议
    print(f"\n💡 建议:")
    if f1 > 0.7:
        print("1. ✅ 当前重新实现版本表现良好，可以继续使用")
        print("2. 🔍 建议扩大测试样本到50-100个视频验证稳定性")
    else:
        print("1. ⚠️ 重新实现版本性能有待提升")
        print("2. 🛠️ 强烈建议修复并使用原始脚本")
        print("3. 🔍 需要详细对比prompt和参数差异")
    
    print("3. 📊 建议进行A/B测试：相同视频用两个版本处理，逐一对比结果")
    print("4. 🌡️ 确保Temperature=0参数在两个版本中都正确设置")
    
    return {
        'f1': f1,
        'precision': precision, 
        'recall': recall,
        'accuracy': accuracy,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'errors': errors,
        'total': len(first_20_results)
    }

if __name__ == "__main__":
    print("🚀 开始快速对比分析...")
    analyze_first_20_videos()