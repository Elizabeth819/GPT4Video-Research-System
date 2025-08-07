#!/usr/bin/env python3
"""
手动生成最终的Gemini分析总结报告
基于run1的完整结果，由于API配额限制无法继续处理失败的视频
"""

import json
import datetime
import os

def generate_final_summary():
    """生成最终总结报告"""
    
    # 读取run1的结果
    run1_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gemini-1.5-flash-run1/gemini_100videos_summary_20250725_211742.json"
    
    with open(run1_file, 'r', encoding='utf-8') as f:
        run1_data = json.load(f)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 统计当前状态
    detailed_results = run1_data["detailed_results"]
    
    # 计算性能指标
    stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "ERROR": 0}
    
    for result in detailed_results:
        stats[result["evaluation"]] += 1
        
    tp, fp, tn, fn = stats["TP"], stats["FP"], stats["TN"], stats["FN"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "stats": stats,
        "total_processed": tp + fp + tn + fn,
        "success_rate": (tp + fp + tn + fn) / len(detailed_results) if len(detailed_results) > 0 else 0
    }
    
    # 创建最终报告
    final_report = {
        "experiment_summary": {
            "timestamp": timestamp,
            "total_videos_attempted": len(detailed_results),
            "successfully_processed": tp + fp + tn + fn,  # 24个
            "failed_due_to_api_limits": stats["ERROR"],  # 74个
            "model": "Gemini-1.5-flash",
            "prompt_version": "balanced_gpt41_style",
            "api_limitation": "Exceeded quota after 24 videos in both API keys",
            "output_directory": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gemini-1.5-flash-run1"
        },
        "performance_metrics": metrics,
        "detailed_results": detailed_results,
        "api_issues": {
            "first_api_key_quota_reached": "After ~24 videos",
            "second_api_key_quota_reached": "Immediately (0 additional videos)",
            "total_quota_errors": stats["ERROR"]
        }
    }
    
    # 保存最终报告
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/"
    final_report_file = os.path.join(output_dir, f"gemini_final_summary_{timestamp}.json")
    with open(final_report_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    
    # 生成markdown报告
    markdown_report = generate_markdown_report(final_report)
    markdown_file = os.path.join(output_dir, f"gemini_final_report_{timestamp}.md")
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    # 打印总结
    print("\\n" + "="*70)
    print("Gemini-1.5-flash 100视频分析最终总结报告")
    print("="*70)
    print(f"计划测试视频数量: 98")
    print(f"成功处理数量: {metrics['total_processed']}")
    print(f"API配额限制失败: {stats['ERROR']}")
    print(f"实际成功率: {metrics['success_rate']:.1%}")
    print(f"测试时间: {timestamp}")
    
    print("\\n基于成功处理的24个视频的性能指标:")
    print(f"  精确度 (Precision): {metrics['precision']:.3f}")
    print(f"  召回率 (Recall): {metrics['recall']:.3f}")
    print(f"  F1分数: {metrics['f1_score']:.3f}")
    print(f"  准确率 (Accuracy): {metrics['accuracy']:.3f}")
    
    print("\\n详细统计:")
    print(f"  True Positives (TP): {stats['TP']}")
    print(f"  False Positives (FP): {stats['FP']}")
    print(f"  True Negatives (TN): {stats['TN']}")
    print(f"  False Negatives (FN): {stats['FN']}")
    print(f"  API配额错误 (ERROR): {stats['ERROR']}")
    
    print("\\nAPI配额限制情况:")
    print(f"  第一个API密钥: 处理24个视频后达到配额限制")
    print(f"  第二个API密钥: 立即达到配额限制 (0个额外视频)")
    print(f"  总配额错误数: {stats['ERROR']}")
    
    print(f"\\n详细结果已保存到:")
    print(f"  JSON报告: {final_report_file}")
    print(f"  Markdown报告: {markdown_file}")
    
    print("\\n结论:")
    print("由于Gemini API配额限制，实验仅完成了24个视频的分析。")
    print(f"基于这24个视频，模型表现为: F1分数 {metrics['f1_score']:.3f}, 精确度 {metrics['precision']:.3f}, 召回率 {metrics['recall']:.3f}")
    print("需要额外的API配额来完成剩余74个视频的分析。")

def generate_markdown_report(report):
    """生成Markdown格式报告"""
    metrics = report["performance_metrics"]
    summary = report["experiment_summary"]
    api_issues = report["api_issues"]
    
    markdown = f"""# Gemini-1.5-flash 100视频Ghost Probing检测最终报告

## 实验概述

- **实验时间**: {summary['timestamp']}
- **模型**: {summary['model']}
- **Prompt版本**: {summary['prompt_version']}
- **计划测试视频数**: {summary['total_videos_attempted']}
- **成功处理数**: {summary['successfully_processed']}
- **API配额限制失败数**: {summary['failed_due_to_api_limits']}
- **实际成功率**: {metrics['success_rate']:.1%}

## API配额限制问题

- **第一个API密钥**: {api_issues['first_api_key_quota_reached']}
- **第二个API密钥**: {api_issues['second_api_key_quota_reached']}
- **总配额错误数**: {api_issues['total_quota_errors']}

## 基于成功处理的{metrics['total_processed']}个视频的性能指标

| 指标 | 数值 |
|------|------|
| 精确度 (Precision) | {metrics['precision']:.3f} |
| 召回率 (Recall) | {metrics['recall']:.3f} |
| F1分数 | {metrics['f1_score']:.3f} |
| 准确率 (Accuracy) | {metrics['accuracy']:.3f} |

## 详细统计

| 分类 | 数量 |
|------|------|
| True Positives (TP) | {metrics['stats']['TP']} |
| False Positives (FP) | {metrics['stats']['FP']} |
| True Negatives (TN) | {metrics['stats']['TN']} |
| False Negatives (FN) | {metrics['stats']['FN']} |
| API配额错误 (ERROR) | {metrics['stats']['ERROR']} |

## 实验限制

本次实验受到Gemini API配额限制的严重影响:

1. **第一个API密钥** (`AIzaSyDCWXFN2MaPaEab8B5dHSiSt9RkVww3AZ8`): 成功处理24个视频后达到配额限制
2. **第二个API密钥** (`AIzaSyA2nNsiLj7MJRSz99w3dtShozrNSBTdHCs`): 立即达到配额限制，无法处理任何额外视频
3. **总计**: 74个视频因为API配额限制无法处理

## 结论

基于成功处理的24个视频，Gemini-1.5-flash模型使用balanced版本的GPT-4.1风格prompt在ghost probing检测任务上的表现为:

- **F1分数**: {metrics['f1_score']:.3f}
- **精确度**: {metrics['precision']:.3f} 
- **召回率**: {metrics['recall']:.3f}

**注意**: 由于API配额限制，这个结果仅基于24/98个视频，可能不能完全代表模型在全部数据集上的真实性能。需要额外的API配额来完成完整的100视频评估。

实验数据保存在: `{summary['output_directory']}`

---
*报告生成时间: {summary['timestamp']}*
*API配额限制导致实验不完整*
"""
    return markdown

if __name__ == "__main__":
    generate_final_summary()