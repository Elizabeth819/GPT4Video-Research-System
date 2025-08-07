#!/usr/bin/env python3
"""
评估简化版LLaVA检测结果
对比ground truth计算准确率、精确率、召回率
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_ground_truth():
    """加载ground truth标签"""
    # 查找labels.csv文件
    possible_paths = [
        "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv",
        "../../result/groundtruth_labels.csv",
        "../groundtruth_labels.csv",
        "./groundtruth_labels.csv"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ 找到ground truth文件: {path}")
            df = pd.read_csv(path, sep='\t')  # 使用tab分隔符
            # 处理标签：包含"ghost probing"的为正样本
            df['ghost_probing_label'] = df['ground_truth_label'].apply(
                lambda x: 1 if 'ghost probing' in str(x) else 0
            )
            return df
    
    print("❌ 未找到ground truth文件")
    return None

def load_simple_results():
    """加载简化版检测结果"""
    json_file = "simple_job_results/artifacts/outputs/results/simple_llava_results_20250721_142913.json"
    
    if not Path(json_file).exists():
        print("❌ 未找到简化版结果文件")
        return None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    for result in data['results']:
        video_id = result['video_id']
        label = 1 if result['ghost_probing_label'] == 'yes' else 0
        confidence = result['confidence']
        
        results.append({
            'video_id': video_id,
            'predicted_label': label,
            'confidence': confidence,
            'ghost_type': result['ghost_type'],
            'processing_time': result['processing_time']
        })
    
    return pd.DataFrame(results)

def extract_video_number(video_id):
    """从video_id提取视频编号"""
    # images_X_YYY -> X_YYY
    parts = video_id.split('_')
    if len(parts) >= 3:
        return f"{parts[1]}_{parts[2]}"
    return video_id

def evaluate_results():
    """评估检测结果"""
    print("🔍 开始评估简化版LLaVA检测结果...")
    print("=" * 60)
    
    # 加载数据
    gt_df = load_ground_truth()
    pred_df = load_simple_results()
    
    if gt_df is None or pred_df is None:
        print("❌ 无法加载数据文件")
        return
    
    print(f"📊 Ground truth数据: {len(gt_df)} 条")
    print(f"📊 预测结果数据: {len(pred_df)} 条")
    
    # 提取视频编号进行匹配
    pred_df['video_number'] = pred_df['video_id'].apply(extract_video_number)
    
    # 匹配数据
    matched_data = []
    for _, pred_row in pred_df.iterrows():
        video_num = pred_row['video_number']
        
        # 在ground truth中查找匹配的视频
        # 构建完整的video_id进行匹配
        full_video_id = f"images_{video_num}.avi"
        gt_match = gt_df[gt_df['video_id'] == full_video_id]
        
        if not gt_match.empty:
            gt_row = gt_match.iloc[0]
            matched_data.append({
                'video_id': pred_row['video_id'],
                'video_number': video_num,
                'predicted_label': pred_row['predicted_label'],
                'true_label': gt_row['ghost_probing_label'],
                'confidence': pred_row['confidence'],
                'ghost_type': pred_row['ghost_type'],
                'processing_time': pred_row['processing_time']
            })
    
    if not matched_data:
        print("❌ 没有找到匹配的数据")
        return
    
    matched_df = pd.DataFrame(matched_data)
    print(f"✅ 成功匹配 {len(matched_df)} 个视频")
    
    # 计算评估指标
    y_true = matched_df['true_label'].values
    y_pred = matched_df['predicted_label'].values
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # 统计信息
    gt_positive = sum(y_true)
    pred_positive = sum(y_pred)
    
    print("\n📊 评估结果:")
    print("=" * 40)
    print(f"总视频数: {len(matched_df)}")
    print(f"Ground Truth正样本: {gt_positive} ({gt_positive/len(matched_df)*100:.1f}%)")
    print(f"预测正样本: {pred_positive} ({pred_positive/len(matched_df)*100:.1f}%)")
    print()
    print("🎯 性能指标:")
    print(f"准确率 (Accuracy): {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"精确率 (Precision): {precision:.3f} ({precision*100:.1f}%)")
    print(f"召回率 (Recall): {recall:.3f} ({recall*100:.1f}%)")
    print(f"F1分数: {f1:.3f}")
    print()
    print("📈 混淆矩阵:")
    print(f"真负例 (TN): {tn}")
    print(f"假正例 (FP): {fp}")
    print(f"假负例 (FN): {fn}")
    print(f"真正例 (TP): {tp}")
    
    # 显示检测到的正样本
    detected_positives = matched_df[matched_df['predicted_label'] == 1]
    if not detected_positives.empty:
        print(f"\n🚨 检测到的鬼探头视频 ({len(detected_positives)} 个):")
        for _, row in detected_positives.iterrows():
            gt_label = "✅" if row['true_label'] == 1 else "❌"
            print(f"  - {row['video_id']} (置信度: {row['confidence']:.3f}) {gt_label}")
    
    # 显示漏检的正样本
    false_negatives = matched_df[(matched_df['true_label'] == 1) & (matched_df['predicted_label'] == 0)]
    if not false_negatives.empty:
        print(f"\n😞 漏检的鬼探头视频 ({len(false_negatives)} 个):")
        for _, row in false_negatives.iterrows():
            print(f"  - {row['video_id']} (置信度: {row['confidence']:.3f})")
    
    # 平均处理时间
    avg_time = matched_df['processing_time'].mean()
    print(f"\n⏱️  平均处理时间: {avg_time:.2f}秒/视频")
    
    # 保存详细结果
    output_file = "simple_llava_evaluation_results.csv"
    matched_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n💾 详细结果已保存到: {output_file}")
    
    # 保存评估报告
    report = {
        'model': 'CLIP-GPT2-Simple',
        'total_videos': len(matched_df),
        'ground_truth_positives': int(gt_positive),
        'predicted_positives': int(pred_positive),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'avg_processing_time': avg_time,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    report_file = "simple_llava_evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 评估报告已保存到: {report_file}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_results()