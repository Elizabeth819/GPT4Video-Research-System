#!/usr/bin/env python3
"""
分析GPT-4.1的鬼探头预测结果
抽样检查预测与Ground Truth标签的对比
"""

import os
import json
import csv
import random

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

def extract_ghost_probing_prediction(result_data):
    """从模型结果中提取ghost probing预测，返回详细信息"""
    if not isinstance(result_data, list):
        return False, []
    
    ghost_evidence = []
    found_ghost = False
    
    # 检查所有段落的分析结果
    for segment in result_data:
        if not isinstance(segment, dict):
            continue
            
        segment_evidence = []
        
        # 检查多个字段中是否提到ghost probing相关内容
        for field in ['summary', 'actions', 'key_actions', 'key_objects']:
            if field in segment and segment[field]:
                text = str(segment[field]).lower()
                
                # 检查是否包含ghost probing相关关键词
                ghost_keywords = [
                    'ghost probing', 'ghost', 'probing', 
                    'sudden appearance', 'unexpected', 'emerging',
                    'appearing suddenly', 'cuts in', 'cut in',
                    'sudden', 'abrupt', 'intrusion'
                ]
                
                for keyword in ghost_keywords:
                    if keyword in text:
                        segment_evidence.append({
                            'field': field,
                            'keyword': keyword,
                            'text': segment[field][:100] + "..." if len(segment[field]) > 100 else segment[field]
                        })
                        found_ghost = True
        
        if segment_evidence:
            ghost_evidence.append({
                'segment_id': segment.get('segment_id', 'unknown'),
                'timestamp': f"{segment.get('Start_Timestamp', 'N/A')} - {segment.get('End_Timestamp', 'N/A')}",
                'evidence': segment_evidence
            })
    
    return found_ghost, ghost_evidence

def analyze_gpt41_predictions():
    """分析GPT-4.1的预测结果"""
    print("🔍 分析GPT-4.1鬼探头预测结果")
    print("=" * 60)
    
    # 加载Ground Truth
    ground_truth = load_ground_truth()
    print(f"📁 加载Ground Truth: {len(ground_truth)} 个标签")
    
    # 获取GPT-4.1处理的视频
    gpt41_dir = "result/gpt41-gt-final"
    gpt41_files = [f for f in os.listdir(gpt41_dir) if f.endswith('.json')]
    
    # 分析所有预测
    predictions = {}
    true_positives = []  # 正确预测的鬼探头
    false_positives = []  # 错误预测的鬼探头
    true_negatives = []  # 正确预测的非鬼探头
    false_negatives = []  # 遗漏的鬼探头
    
    for filename in gpt41_files:
        video_id = filename.replace('actionSummary_', '').replace('.json', '')
        
        if video_id not in ground_truth:
            continue
            
        # 加载GPT-4.1结果
        with open(os.path.join(gpt41_dir, filename), 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # 提取预测
        predicted_ghost, evidence = extract_ghost_probing_prediction(result_data)
        
        # 获取Ground Truth标签
        gt_label = ground_truth[video_id]
        is_ghost_in_gt = 'ghost probing' in gt_label.lower()
        
        predictions[video_id] = {
            'predicted': predicted_ghost,
            'ground_truth': is_ghost_in_gt,
            'gt_label': gt_label,
            'evidence': evidence
        }
        
        # 分类结果
        if predicted_ghost and is_ghost_in_gt:
            true_positives.append(video_id)
        elif predicted_ghost and not is_ghost_in_gt:
            false_positives.append(video_id)
        elif not predicted_ghost and not is_ghost_in_gt:
            true_negatives.append(video_id)
        elif not predicted_ghost and is_ghost_in_gt:
            false_negatives.append(video_id)
    
    # 计算统计
    total = len(predictions)
    tp_count = len(true_positives)
    fp_count = len(false_positives) 
    tn_count = len(true_negatives)
    fn_count = len(false_negatives)
    
    accuracy = (tp_count + tn_count) / total if total > 0 else 0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 整体统计:")
    print(f"   总视频数: {total}")
    print(f"   真正例 (TP): {tp_count}")
    print(f"   假正例 (FP): {fp_count}")
    print(f"   真负例 (TN): {tn_count}")
    print(f"   假负例 (FN): {fn_count}")
    print(f"   准确率: {accuracy:.4f}")
    print(f"   精确度: {precision:.4f}")
    print(f"   召回率: {recall:.4f}")
    print(f"   F1分数: {f1:.4f}")
    
    return predictions, true_positives, false_positives, true_negatives, false_negatives

def show_sample_analysis(predictions, true_positives, false_positives):
    """展示样本分析"""
    print(f"\n" + "="*80)
    print("🎯 样本分析：GPT-4.1预测 vs Ground Truth标签")
    print("="*80)
    
    # 1. 展示正确预测的鬼探头样本
    print(f"\n✅ 正确预测的鬼探头样本 (TP) - 抽样5个:")
    print("-" * 60)
    
    tp_sample = random.sample(true_positives, min(5, len(true_positives)))
    for i, video_id in enumerate(tp_sample, 1):
        pred = predictions[video_id]
        print(f"\n{i}. 视频: {video_id}")
        print(f"   Ground Truth: {pred['gt_label']}")
        print(f"   GPT-4.1预测: 鬼探头 ✓")
        
        if pred['evidence']:
            print(f"   GPT-4.1证据:")
            for segment in pred['evidence'][:2]:  # 只显示前2个segment
                print(f"     时间段 {segment['timestamp']}:")
                for evidence in segment['evidence'][:2]:  # 只显示前2个证据
                    print(f"       {evidence['field']}: \"{evidence['text']}\"")
    
    # 2. 展示错误预测的鬼探头样本
    print(f"\n❌ 错误预测的鬼探头样本 (FP) - 抽样5个:")
    print("-" * 60)
    
    fp_sample = random.sample(false_positives, min(5, len(false_positives)))
    for i, video_id in enumerate(fp_sample, 1):
        pred = predictions[video_id]
        print(f"\n{i}. 视频: {video_id}")
        print(f"   Ground Truth: {pred['gt_label']}")
        print(f"   GPT-4.1预测: 鬼探头 ❌ (误报)")
        
        if pred['evidence']:
            print(f"   GPT-4.1错误证据:")
            for segment in pred['evidence'][:2]:
                print(f"     时间段 {segment['timestamp']}:")
                for evidence in segment['evidence'][:2]:
                    print(f"       {evidence['field']}: \"{evidence['text']}\"")

def analyze_precision_issues(false_positives, predictions):
    """分析精确度不高的原因"""
    print(f"\n" + "="*80)
    print("🔍 精确度分析：为什么GPT-4.1精确度不高？")
    print("="*80)
    
    print(f"📊 误报统计:")
    print(f"   总误报数: {len(false_positives)}")
    print(f"   误报率: {len(false_positives)/(len(false_positives) + len([v for v in predictions.values() if v['predicted'] and v['ground_truth']]))*100:.1f}%")
    
    # 分析误报的关键词模式
    keyword_counts = {}
    trigger_patterns = {}
    
    for video_id in false_positives:
        pred = predictions[video_id]
        for segment in pred['evidence']:
            for evidence in segment['evidence']:
                keyword = evidence['keyword']
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                
                # 记录触发模式
                text = evidence['text'].lower()
                if keyword not in trigger_patterns:
                    trigger_patterns[keyword] = []
                trigger_patterns[keyword].append(text)
    
    print(f"\n📈 导致误报的关键词频率:")
    for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   '{keyword}': {count} 次")
    
    print(f"\n🔍 误报原因分析:")
    
    # 分析常见误报模式
    common_false_triggers = []
    
    for video_id in false_positives[:10]:  # 分析前10个误报
        pred = predictions[video_id]
        gt_label = pred['gt_label']
        
        # 分析为什么被误报
        if 'none' in gt_label.lower():
            reason = "Ground Truth标注为'none'，但GPT-4.1检测到了可能的危险行为"
        elif any(word in gt_label.lower() for word in ['normal', 'safe', 'routine']):
            reason = "Ground Truth标注为正常场景，但GPT-4.1过度敏感"
        elif any(word in gt_label.lower() for word in ['overtaking', 'lane change', 'turn']):
            reason = "正常的超车/变道被误识别为鬼探头"
        else:
            reason = "其他原因"
        
        common_false_triggers.append({
            'video_id': video_id,
            'gt_label': gt_label,
            'reason': reason
        })
    
    # 统计误报原因
    reason_counts = {}
    for item in common_false_triggers:
        reason = item['reason']
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    print(f"\n📋 主要误报原因:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {reason}: {count} 个案例")
    
    print(f"\n💡 改进建议:")
    print(f"   1. 调整关键词敏感度，减少对'sudden'、'abrupt'等常见词的过度反应")
    print(f"   2. 增加上下文理解，区分正常行驶行为和真正的鬼探头")
    print(f"   3. 添加时间和空间约束，鬼探头通常涉及非常接近的距离和突然性")
    print(f"   4. 结合多个时间段的信息，避免单一帧的误判")

def main():
    print("🔍 GPT-4.1鬼探头预测分析")
    print("=" * 60)
    
    # 设置随机种子以确保结果可重现
    random.seed(42)
    
    # 分析预测结果
    predictions, tp, fp, tn, fn = analyze_gpt41_predictions()
    
    # 展示样本分析
    show_sample_analysis(predictions, tp, fp)
    
    # 分析精确度问题
    analyze_precision_issues(fp, predictions)
    
    print(f"\n" + "="*80)
    print("📊 总结:")
    print(f"   GPT-4.1在召回率方面表现优秀 (遗漏很少)")
    print(f"   但在精确度方面需要改进 (误报较多)")
    print(f"   这是一个典型的高敏感度模型特征")
    print("="*80)

if __name__ == "__main__":
    main()