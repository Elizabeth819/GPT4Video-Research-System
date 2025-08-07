#!/usr/bin/env python3
"""
原始脚本 vs 重新实现版本对比分析
使用相同的20个视频比较两个版本的准确率差异
"""

import os
import json
import pandas as pd
import datetime
from collections import Counter

def load_ground_truth():
    """加载ground truth"""
    gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/groundtruth_labels.csv"
    return pd.read_csv(gt_path, sep='\t')

def load_original_results():
    """加载原始脚本结果"""
    results_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/original_script_results"
    original_results = []
    
    test_videos = [
        "images_1_001.avi", "images_1_002.avi", "images_1_003.avi", "images_1_004.avi", "images_1_005.avi",
        "images_1_006.avi", "images_1_007.avi", "images_1_008.avi", "images_1_009.avi", "images_1_010.avi",
        "images_1_011.avi", "images_1_012.avi", "images_1_013.avi", "images_1_014.avi", "images_1_015.avi",
        "images_1_016.avi", "images_1_017.avi", "images_1_018.avi", "images_1_019.avi", "images_1_020.avi"
    ]
    
    for video_id in test_videos:
        result_file = os.path.join(results_dir, f"actionSummary_{video_id.split('.')[0]}.json")
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 原始脚本返回间隔列表，需要整合key_actions
                key_actions_list = []
                if isinstance(data, list):
                    for interval in data:
                        if 'key_actions' in interval:
                            ka = interval['key_actions'].lower()
                            if ka and ka != 'none':
                                key_actions_list.append(ka)
                
                # 整合key_actions：如果有任何interval检测到ghost probing，则认为整个视频有
                if any('ghost probing' in ka for ka in key_actions_list):
                    final_key_actions = 'ghost probing'
                elif any('cut-in' in ka for ka in key_actions_list):
                    final_key_actions = 'cut-in'  
                elif any('overtaking' in ka for ka in key_actions_list):
                    final_key_actions = 'overtaking'
                else:
                    final_key_actions = 'none'
                    
                original_results.append({
                    "video_id": video_id,
                    "key_actions": final_key_actions,
                    "raw_data": data
                })
                
            except Exception as e:
                print(f"解析原始结果失败 {video_id}: {str(e)}")
                
    return original_results

def load_run7_results():
    """加载Run 7的结果"""
    run7_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run7-gpt4o-paper-batch-temp0"
    
    # 找到最新的中间结果文件
    files = [f for f in os.listdir(run7_dir) if f.startswith("run7_intermediate_")]
    if not files:
        return []
        
    latest_file = sorted(files)[-1]
    file_path = os.path.join(run7_dir, latest_file)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        run7_data = json.load(f)
    
    # 提取前20个视频的结果
    first_20_videos = [
        "images_1_001.avi", "images_1_002.avi", "images_1_003.avi", "images_1_004.avi", "images_1_005.avi",
        "images_1_006.avi", "images_1_007.avi", "images_1_008.avi", "images_1_009.avi", "images_1_010.avi",
        "images_1_011.avi", "images_1_012.avi", "images_1_013.avi", "images_1_014.avi", "images_1_015.avi",
        "images_1_016.avi", "images_1_017.avi", "images_1_018.avi", "images_1_019.avi", "images_1_020.avi"
    ]
    
    run7_results = []
    for result in run7_data["detailed_results"]:
        if result["video_id"] in first_20_videos:
            run7_results.append({
                "video_id": result["video_id"],
                "key_actions": result["key_actions"].lower(),
                "evaluation": result["evaluation"]
            })
    
    # 按照顺序排序
    run7_results.sort(key=lambda x: first_20_videos.index(x["video_id"]))
    return run7_results

def evaluate_result(video_id, key_actions, ground_truth_label):
    """评估结果"""
    has_ghost_probing = "ghost probing" in key_actions
    ground_truth_has_ghost = ground_truth_label != "none"
    
    if has_ghost_probing and ground_truth_has_ghost:
        return "TP"
    elif has_ghost_probing and not ground_truth_has_ghost:
        return "FP"
    elif not has_ghost_probing and ground_truth_has_ghost:
        return "FN"
    else:
        return "TN"

def calculate_metrics(results):
    """计算性能指标"""
    evals = [r['evaluation'] for r in results]
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
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'errors': errors,
        'total': len(results)
    }

def main():
    print("🔍 开始原始脚本 vs 重新实现版本对比分析...")
    print("=" * 80)
    
    # 加载数据
    ground_truth = load_ground_truth()
    original_results = load_original_results()
    run7_results = load_run7_results()
    
    print(f"📊 数据加载完成:")
    print(f"   原始脚本结果: {len(original_results)} 个视频")
    print(f"   Run 7结果: {len(run7_results)} 个视频")
    print(f"   Ground Truth: {len(ground_truth)} 个标签")
    
    # 为原始脚本结果添加评估
    for result in original_results:
        video_id = result["video_id"]
        key_actions = result["key_actions"]
        
        gt_row = ground_truth[ground_truth['video_id'] == video_id]
        if not gt_row.empty:
            ground_truth_label = gt_row.iloc[0]['ground_truth_label']
            evaluation = evaluate_result(video_id, key_actions, ground_truth_label)
            result['ground_truth'] = ground_truth_label
            result['evaluation'] = evaluation
        else:
            result['ground_truth'] = 'unknown'
            result['evaluation'] = 'ERROR'
    
    # 计算指标
    original_metrics = calculate_metrics(original_results)
    
    # 从run7结果提取指标
    run7_metrics = calculate_metrics(run7_results)
    
    # 生成对比报告
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n📈 性能对比结果:")
    print("=" * 80)
    
    print(f"\n🔴 原始脚本版本 (paper_cleanup_backup):")
    print(f"   精确度: {original_metrics['precision']:.3f} ({original_metrics['precision']*100:.1f}%)")
    print(f"   召回率: {original_metrics['recall']:.3f} ({original_metrics['recall']*100:.1f}%)")
    print(f"   F1分数: {original_metrics['f1']:.3f} ({original_metrics['f1']*100:.1f}%)")
    print(f"   准确率: {original_metrics['accuracy']:.3f} ({original_metrics['accuracy']*100:.1f}%)")
    print(f"   TP: {original_metrics['tp']}, FP: {original_metrics['fp']}, TN: {original_metrics['tn']}, FN: {original_metrics['fn']}")
    
    print(f"\n🔵 重新实现版本 (Run 7):")
    print(f"   精确度: {run7_metrics['precision']:.3f} ({run7_metrics['precision']*100:.1f}%)")
    print(f"   召回率: {run7_metrics['recall']:.3f} ({run7_metrics['recall']*100:.1f}%)")
    print(f"   F1分数: {run7_metrics['f1']:.3f} ({run7_metrics['f1']*100:.1f}%)")
    print(f"   准确率: {run7_metrics['accuracy']:.3f} ({run7_metrics['accuracy']*100:.1f}%)")
    print(f"   TP: {run7_metrics['tp']}, FP: {run7_metrics['fp']}, TN: {run7_metrics['tn']}, FN: {run7_metrics['fn']}")
    
    print(f"\n📊 版本差异:")
    print(f"   F1分数差异: {original_metrics['f1'] - run7_metrics['f1']:+.3f} (原始 - 重新实现)")
    print(f"   精确度差异: {original_metrics['precision'] - run7_metrics['precision']:+.3f}")
    print(f"   召回率差异: {original_metrics['recall'] - run7_metrics['recall']:+.3f}")
    
    # 逐视频对比
    print(f"\n🔍 逐视频对比分析:")
    print("=" * 80)
    print("视频ID\t\t\tGround Truth\t原始脚本\t\tRun 7\t\t一致性")
    print("-" * 80)
    
    agreements = 0
    disagreements = []
    
    for orig in original_results:
        video_id = orig["video_id"]
        orig_ka = orig["key_actions"]
        orig_eval = orig["evaluation"]
        gt = orig["ground_truth"]
        
        # 找到对应的Run 7结果
        run7_result = next((r for r in run7_results if r["video_id"] == video_id), None)
        if run7_result:
            run7_ka = run7_result["key_actions"]
            run7_eval = run7_result["evaluation"]
            
            # 检查一致性
            consistent = orig_eval == run7_eval
            if consistent:
                agreements += 1
            else:
                disagreements.append({
                    'video_id': video_id,
                    'ground_truth': gt,
                    'original': orig_eval,
                    'run7': run7_eval,
                    'original_ka': orig_ka,
                    'run7_ka': run7_ka
                })
            
            consistency_symbol = "✅" if consistent else "❌"
            print(f"{video_id}\t{gt:<10}\t{orig_eval:<4}({orig_ka[:10]}...)\t{run7_eval:<4}({run7_ka[:10]}...)\t{consistency_symbol}")
    
    print(f"\n📈 一致性统计:")
    total_videos = len(original_results)
    consistency_rate = agreements / total_videos
    print(f"   一致视频数: {agreements}/{total_videos}")
    print(f"   一致性率: {consistency_rate:.1%}")
    
    if disagreements:
        print(f"\n❌ 不一致案例分析 ({len(disagreements)} 个):")
        for i, case in enumerate(disagreements, 1):
            print(f"   {i}. {case['video_id']}: GT={case['ground_truth']}")
            print(f"      原始脚本: {case['original']} (key_actions: {case['original_ka']})")
            print(f"      Run 7: {case['run7']} (key_actions: {case['run7_ka']})")
    
    # 结论
    print(f"\n🎯 结论和建议:")
    print("=" * 80)
    
    if original_metrics['f1'] > run7_metrics['f1']:
        print("✅ 原始脚本版本表现更优")
        print(f"   F1分数高出 {(original_metrics['f1'] - run7_metrics['f1'])*100:.1f} 个百分点")
        print("💡 建议: 使用修复后的原始脚本进行后续实验")
    elif run7_metrics['f1'] > original_metrics['f1']:
        print("✅ 重新实现版本表现更优")
        print(f"   F1分数高出 {(run7_metrics['f1'] - original_metrics['f1'])*100:.1f} 个百分点")
        print("💡 建议: 继续使用重新实现版本")
    else:
        print("⚖️ 两个版本表现相当")
        print("💡 建议: 可以选择任一版本，优先使用原始脚本")
    
    print(f"\n🔧 技术分析:")
    if consistency_rate < 0.8:
        print(f"⚠️  一致性率较低 ({consistency_rate:.1%})，可能存在:")
        print("   - Prompt实现差异")
        print("   - 图像处理流程差异") 
        print("   - API调用参数差异")
        print("   - Temperature参数设置差异")
    else:
        print(f"✅ 一致性率良好 ({consistency_rate:.1%})，两个版本基本等价")
    
    print(f"\n📝 测试报告生成时间: {timestamp}")
    print("🔚 对比分析完成!")

if __name__ == "__main__":
    main()