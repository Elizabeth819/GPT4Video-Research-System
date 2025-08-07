#!/usr/bin/env python3
"""
全面对比分析：原始脚本 vs Run 7 vs Run 7 Enhanced (Few-shot)
比较三个版本在相同20个视频上的性能差异
"""

import json
import pandas as pd
import datetime
from collections import Counter
import os

def load_results():
    """加载所有版本的结果"""
    results = {}
    
    # 1. 原始脚本结果 (已经从之前的对比中获得)
    original_results = [
        {"video_id": "images_1_001.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_002.avi", "key_actions": "none", "evaluation": "FN", "ground_truth": "5s: ghost probing"},
        {"video_id": "images_1_003.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "2s: ghost probing"},
        {"video_id": "images_1_004.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_005.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "8s: ghost probing"},
        {"video_id": "images_1_006.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "9s: ghost probing"},
        {"video_id": "images_1_007.avi", "key_actions": "none", "evaluation": "FN", "ground_truth": "6s: ghost probing"},
        {"video_id": "images_1_008.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "3s: ghost probing"},
        {"video_id": "images_1_009.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_010.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "15s: ghost probing"},
        {"video_id": "images_1_011.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "11s: ghost probing"},
        {"video_id": "images_1_012.avi", "key_actions": "cut-in", "evaluation": "FN", "ground_truth": "11s: ghost probing"},
        {"video_id": "images_1_013.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "8s: ghost probing"},
        {"video_id": "images_1_014.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "5s: ghost probing"},
        {"video_id": "images_1_015.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "5s: ghost probing"},
        {"video_id": "images_1_016.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "4s: ghost probing"},
        {"video_id": "images_1_017.avi", "key_actions": "cut-in", "evaluation": "FN", "ground_truth": "17s: ghost probing"},
        {"video_id": "images_1_018.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_019.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_020.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"}
    ]
    results["original"] = original_results
    
    # 2. Run 7结果 (从之前的对比中获得)
    run7_results = [
        {"video_id": "images_1_001.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_002.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "5s: ghost probing"},
        {"video_id": "images_1_003.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "2s: ghost probing"},
        {"video_id": "images_1_004.avi", "key_actions": "none", "evaluation": "TN", "ground_truth": "none"},
        {"video_id": "images_1_005.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "8s: ghost probing"},
        {"video_id": "images_1_006.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "9s: ghost probing"},
        {"video_id": "images_1_007.avi", "key_actions": "none", "evaluation": "FN", "ground_truth": "6s: ghost probing"},
        {"video_id": "images_1_008.avi", "key_actions": "none", "evaluation": "FN", "ground_truth": "3s: ghost probing"},
        {"video_id": "images_1_009.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_010.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "15s: ghost probing"},
        {"video_id": "images_1_011.avi", "key_actions": "cut-in", "evaluation": "FN", "ground_truth": "11s: ghost probing"},
        {"video_id": "images_1_012.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "11s: ghost probing"},
        {"video_id": "images_1_013.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "8s: ghost probing"},
        {"video_id": "images_1_014.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "5s: ghost probing"},
        {"video_id": "images_1_015.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "5s: ghost probing"},
        {"video_id": "images_1_016.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "4s: ghost probing"},
        {"video_id": "images_1_017.avi", "key_actions": "", "evaluation": "ERROR", "ground_truth": "17s: ghost probing"},
        {"video_id": "images_1_018.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_019.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_020.avi", "key_actions": "none", "evaluation": "TN", "ground_truth": "none"}
    ]
    results["run7"] = run7_results
    
    # 3. Run 7 Enhanced (Few-shot) 结果 (从刚才的实验获得)
    enhanced_results = [
        {"video_id": "images_1_001.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_002.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "5s: ghost probing"},
        {"video_id": "images_1_003.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "2s: ghost probing"},
        {"video_id": "images_1_004.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_005.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "8s: ghost probing"},
        {"video_id": "images_1_006.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "9s: ghost probing"},
        {"video_id": "images_1_007.avi", "key_actions": "none", "evaluation": "FN", "ground_truth": "6s: ghost probing"},
        {"video_id": "images_1_008.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "3s: ghost probing"},
        {"video_id": "images_1_009.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_010.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "15s: ghost probing"},
        {"video_id": "images_1_011.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "11s: ghost probing"},
        {"video_id": "images_1_012.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "11s: ghost probing"},
        {"video_id": "images_1_013.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "8s: ghost probing"},
        {"video_id": "images_1_014.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "5s: ghost probing"},
        {"video_id": "images_1_015.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "5s: ghost probing"},
        {"video_id": "images_1_016.avi", "key_actions": "ghost probing", "evaluation": "TP", "ground_truth": "4s: ghost probing"},
        {"video_id": "images_1_017.avi", "key_actions": "none", "evaluation": "FN", "ground_truth": "17s: ghost probing"},
        {"video_id": "images_1_018.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_019.avi", "key_actions": "ghost probing", "evaluation": "FP", "ground_truth": "none"},
        {"video_id": "images_1_020.avi", "key_actions": "none", "evaluation": "TN", "ground_truth": "none"}
    ]
    results["enhanced"] = enhanced_results
    
    return results

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

def analyze_differences(results_dict):
    """分析版本间差异"""
    print("🔍 逐视频详细对比分析:")
    print("=" * 100)
    print(f"{'视频ID':<20} {'Ground Truth':<15} {'原始脚本':<10} {'Run 7':<10} {'Enhanced':<10} {'差异说明'}")
    print("-" * 100)
    
    differences = []
    for i in range(20):
        video_id = results_dict["original"][i]["video_id"]
        gt = results_dict["original"][i]["ground_truth"]
        
        orig_eval = results_dict["original"][i]["evaluation"]
        run7_eval = results_dict["run7"][i]["evaluation"]
        enhanced_eval = results_dict["enhanced"][i]["evaluation"]
        
        # 分析差异
        diff_desc = ""
        if orig_eval != run7_eval or run7_eval != enhanced_eval or orig_eval != enhanced_eval:
            improvements = []
            if enhanced_eval == "TP" and run7_eval != "TP":
                improvements.append("Enhanced修复了Run7的漏检")
            if enhanced_eval == "TN" and run7_eval != "TN":
                improvements.append("Enhanced减少了误报")
            if run7_eval == "TP" and orig_eval != "TP":
                improvements.append("Run7修复了原始脚本问题")
            if enhanced_eval == "TP" and orig_eval != "TP":
                improvements.append("Enhanced修复了原始脚本问题")
                
            if improvements:
                diff_desc = "; ".join(improvements)
            else:
                diff_desc = "存在性能差异"
                
            differences.append({
                'video_id': video_id,
                'ground_truth': gt,
                'original': orig_eval,
                'run7': run7_eval, 
                'enhanced': enhanced_eval,
                'difference': diff_desc
            })
        
        print(f"{video_id:<20} {gt:<15} {orig_eval:<10} {run7_eval:<10} {enhanced_eval:<10} {diff_desc}")
    
    return differences

def main():
    print("🚀 开始全面对比分析...")
    print("=" * 80)
    
    # 加载所有结果
    results_dict = load_results()
    
    # 计算各版本指标
    metrics = {}
    for version, results in results_dict.items():
        metrics[version] = calculate_metrics(results)
    
    # 输出性能对比
    print(f"\n📊 三版本性能对比:")
    print("=" * 80)
    print(f"{'指标':<12} {'原始脚本':<12} {'Run 7':<12} {'Enhanced':<12} {'最佳版本'}")
    print("-" * 60)
    
    for metric in ['f1', 'precision', 'recall', 'accuracy']:
        orig_val = metrics['original'][metric]
        run7_val = metrics['run7'][metric]
        enhanced_val = metrics['enhanced'][metric]
        
        best = max(orig_val, run7_val, enhanced_val)
        best_version = ""
        if best == enhanced_val:
            best_version = "Enhanced"
        elif best == run7_val:
            best_version = "Run 7"
        else:
            best_version = "Original"
            
        print(f"{metric.upper():<12} {orig_val:<12.3f} {run7_val:<12.3f} {enhanced_val:<12.3f} {best_version}")
    
    print(f"\n📈 混淆矩阵对比:")
    print("-" * 60)
    print(f"{'版本':<12} {'TP':<4} {'FP':<4} {'TN':<4} {'FN':<4} {'ERROR':<6}")
    print("-" * 40)
    for version in ['original', 'run7', 'enhanced']:
        m = metrics[version]
        print(f"{version:<12} {m['tp']:<4} {m['fp']:<4} {m['tn']:<4} {m['fn']:<4} {m['errors']:<6}")
    
    # 分析差异
    differences = analyze_differences(results_dict)
    
    print(f"\n🎯 关键发现:")
    print("=" * 80)
    
    # F1分数对比
    orig_f1 = metrics['original']['f1']
    run7_f1 = metrics['run7']['f1']
    enhanced_f1 = metrics['enhanced']['f1']
    
    print(f"1. **F1分数提升轨迹**:")
    print(f"   原始脚本: {orig_f1:.3f} → Run 7: {run7_f1:.3f} → Enhanced: {enhanced_f1:.3f}")
    print(f"   总体提升: {enhanced_f1 - orig_f1:+.3f} ({(enhanced_f1 - orig_f1)*100:+.1f}%)")
    print(f"   Few-shot贡献: {enhanced_f1 - run7_f1:+.3f} ({(enhanced_f1 - run7_f1)*100:+.1f}%)")
    
    print(f"\n2. **Few-shot Examples的影响**:")
    if enhanced_f1 > run7_f1:
        print(f"   ✅ Few-shot examples带来了 {(enhanced_f1 - run7_f1)*100:.1f}% 的F1提升")
        print(f"   ✅ 主要改进: 减少了漏检，提高了召回率")
    else:
        print(f"   ⚠️ Few-shot examples未带来F1提升，可能过拟合")
    
    # 召回率分析
    orig_recall = metrics['original']['recall']
    run7_recall = metrics['run7']['recall']
    enhanced_recall = metrics['enhanced']['recall']
    
    print(f"\n3. **召回率改进分析**:")
    print(f"   原始脚本: {orig_recall:.3f} → Enhanced: {enhanced_recall:.3f} (提升 {(enhanced_recall-orig_recall)*100:+.1f}%)")
    print(f"   Few-shot对召回率的贡献: {(enhanced_recall-run7_recall)*100:+.1f}%")
    
    # 精确度分析
    orig_precision = metrics['original']['precision']
    run7_precision = metrics['run7']['precision']
    enhanced_precision = metrics['enhanced']['precision']
    
    print(f"\n4. **精确度变化分析**:")
    print(f"   原始脚本: {orig_precision:.3f} → Enhanced: {enhanced_precision:.3f} (变化 {(enhanced_precision-orig_precision)*100:+.1f}%)")
    if enhanced_precision < orig_precision:
        print(f"   ⚠️ 精确度有所下降，但召回率大幅提升，整体F1仍然最优")
    
    print(f"\n5. **关键改进案例**:")
    key_improvements = []
    for diff in differences:
        if "Enhanced修复" in diff['difference']:
            key_improvements.append(f"   • {diff['video_id']}: {diff['difference']}")
    
    for improvement in key_improvements[:5]:  # 显示前5个改进案例
        print(improvement)
    
    print(f"\n💡 **结论和建议**:")
    print("=" * 80)
    
    if enhanced_f1 > max(orig_f1, run7_f1):
        print("✅ **Run 7 Enhanced (Few-shot)版本表现最佳**")
        print(f"   - F1分数: {enhanced_f1:.3f} (提升 {(enhanced_f1-orig_f1)*100:.1f}%)")
        print(f"   - 召回率: {enhanced_recall:.3f} (更好的漏检控制)")
        print(f"   - Few-shot examples提供了有效的引导")
        print(f"\n🚀 **推荐**: 使用Enhanced版本进行大规模实验")
    else:
        print("📊 **性能对比结果复杂**")
        print("   需要根据具体应用场景选择合适版本")
    
    print(f"\n📝 **技术洞察**:")
    print("1. **架构简化** (原始→Run7): 主要性能提升来源")
    print("2. **Few-shot Learning**: 进一步优化了边界案例检测") 
    print("3. **Temperature=0**: 确保了输出一致性")
    print("4. **Cut-in移除**: 减少了分类干扰")
    
    # 保存详细报告
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/VIP/comprehensive_comparison_report_{timestamp}.json"
    
    full_report = {
        "timestamp": timestamp,
        "metrics": metrics,
        "differences": differences,
        "summary": {
            "best_version": "enhanced" if enhanced_f1 > max(orig_f1, run7_f1) else "run7",
            "f1_improvements": {
                "original_to_run7": run7_f1 - orig_f1,
                "run7_to_enhanced": enhanced_f1 - run7_f1,
                "total_improvement": enhanced_f1 - orig_f1
            }
        }
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 **详细报告已保存**: {report_file}")
    print(f"🏁 **对比分析完成**!")

if __name__ == "__main__":
    main()