#!/usr/bin/env python3
"""
创建实际的三模型对比报告
基于可用的处理结果进行评估
"""

import os
import json
import csv
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

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
                
                # 解析标签：检查是否包含ghost probing
                if 'ghost probing' in label.lower():
                    ground_truth[video_id] = 1  # 正例
                else:
                    ground_truth[video_id] = 0  # 负例
    
    return ground_truth

def extract_ghost_probing_prediction(result_data):
    """从模型结果中提取ghost probing预测"""
    if not isinstance(result_data, list):
        return 0
    
    # 检查所有段落的分析结果
    for segment in result_data:
        if not isinstance(segment, dict):
            continue
            
        # 检查多个字段中是否提到ghost probing相关内容
        text_fields = []
        for field in ['summary', 'actions', 'key_actions', 'next_action', 'key_objects']:
            if field in segment and segment[field]:
                text_fields.append(str(segment[field]).lower())
        
        combined_text = ' '.join(text_fields)
        
        # 检查是否包含ghost probing相关关键词
        ghost_keywords = [
            'ghost probing', 'ghost', 'probing', 
            'sudden appearance', 'unexpected', 'emerging',
            'appearing suddenly', 'cuts in', 'cut in',
            'overtaking', 'lane change', 'dangerous',
            'risky maneuver', 'unsafe', 'sudden',
            'abrupt', 'intrusion', 'interference',
            'cuts into', 'merging aggressively'
        ]
        
        for keyword in ghost_keywords:
            if keyword in combined_text:
                return 1  # 预测为正例
    
    return 0  # 预测为负例

def find_available_models():
    """查找可用的模型结果目录"""
    potential_dirs = {
        'GPT-4o': ['result/gpt4o-100-3rd', 'result/gpt4o-gt'],
        'GPT-4.1': ['result/gpt41-gt-final', 'result/gpt41-gt-complete', 'result/gpt41-format-test'],
        'Gemini': ['result/gemini-1.5-flash', 'result/gemini-gt-test']
    }
    
    available_models = {}
    for model_name, dirs in potential_dirs.items():
        for dir_path in dirs:
            if os.path.exists(dir_path):
                # 检查目录中是否有JSON文件
                json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
                if json_files:
                    available_models[model_name] = dir_path
                    break
    
    return available_models

def evaluate_model(model_name, result_dir, ground_truth):
    """评估单个模型的性能"""
    print(f"   🔍 评估 {model_name} (目录: {result_dir})")
    
    # 获取处理的视频文件
    result_files = [f for f in os.listdir(result_dir) if f.endswith('.json')]
    processed_videos = []
    
    for filename in result_files:
        video_id = filename.replace('actionSummary_', '').replace('.json', '')
        if video_id in ground_truth:
            processed_videos.append(video_id)
    
    print(f"      找到 {len(processed_videos)} 个Ground Truth视频")
    
    if len(processed_videos) == 0:
        print(f"      ⚠️ 没有处理Ground Truth视频")
        return None
    
    # 评估预测结果
    predictions = {}
    failed_loads = 0
    
    for video_id in processed_videos:
        result_file = os.path.join(result_dir, f"actionSummary_{video_id}.json")
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
                predictions[video_id] = extract_ghost_probing_prediction(result_data)
        except Exception as e:
            print(f"      ⚠️ 加载{video_id}失败: {str(e)[:50]}...")
            predictions[video_id] = 0
            failed_loads += 1
    
    if failed_loads > 0:
        print(f"      ⚠️ {failed_loads} 个文件加载失败")
    
    # 计算指标
    y_true = [ground_truth[video_id] for video_id in processed_videos]
    y_pred = [predictions[video_id] for video_id in processed_videos]
    
    # 检查是否有足够的数据进行评估
    if len(set(y_true)) < 2 or len(set(y_pred)) < 1:
        print(f"      ⚠️ 数据不足，无法计算完整指标")
        return {
            'model': model_name,
            'sample_size': len(processed_videos),
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'specificity': 0,
            'balanced_accuracy': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'note': 'Insufficient data for evaluation'
        }
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # 处理只有一个类别的情况
        if len(set(y_true)) == 1:
            if y_true[0] == 0:  # 只有负例
                tn = sum([1 for p in y_pred if p == 0])
                fp = sum([1 for p in y_pred if p == 1])
                fn = tp = 0
            else:  # 只有正例
                tp = sum([1 for p in y_pred if p == 1])
                fn = sum([1 for p in y_pred if p == 0])
                tn = fp = 0
        else:
            tn = fp = fn = tp = 0
    
    # 特异性和平衡准确率
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    return {
        'model': model_name,
        'sample_size': len(processed_videos),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'processed_videos': processed_videos
    }

def create_practical_comparison_report():
    """创建实际的三模型对比报告"""
    print("📊 创建实际三模型对比报告")
    print("=" * 60)
    
    # 加载Ground Truth
    ground_truth = load_ground_truth()
    print(f"📁 加载Ground Truth: {len(ground_truth)} 个视频")
    
    # 查找可用模型
    available_models = find_available_models()
    print(f"🔍 发现可用模型: {list(available_models.keys())}")
    
    if not available_models:
        print("❌ 没有找到可用的模型结果")
        return None
    
    # 评估所有可用模型
    model_results = {}
    for model_name, result_dir in available_models.items():
        print(f"\n🔍 评估 {model_name}...")
        metrics = evaluate_model(model_name, result_dir, ground_truth)
        if metrics:
            model_results[model_name.lower().replace('-', '').replace('.', '')] = metrics
            print(f"   ✅ {model_name} 评估完成")
        else:
            print(f"   ❌ {model_name} 评估失败")
    
    if not model_results:
        print("❌ 没有可用的模型评估结果")
        return None
    
    # 创建报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "description": "Practical comparison based on available model results",
        "ground_truth_info": {
            "total_gt_samples": len(ground_truth),
            "positive_samples": sum(ground_truth.values()),
            "negative_samples": len(ground_truth) - sum(ground_truth.values())
        },
        "model_results": model_results,
        "available_models": available_models,
        "evaluation_notes": {
            "note": "Evaluation based on actually processed videos by each model",
            "gpt41_limitation": "GPT-4.1 processing limited by Azure content filtering policies"
        }
    }
    
    # 确保比较目录存在
    os.makedirs("result/comparison", exist_ok=True)
    
    # 保存报告
    report_file = f"result/comparison/practical_three_models_comparison_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 报告保存到: {report_file}")
    
    return report

def print_practical_comparison_summary(report):
    """打印实际对比总结"""
    print("\n" + "="*100)
    print("🏆 GPT-4o vs GPT-4.1 vs Gemini 实际对比结果")
    print("基于各模型实际处理的Ground Truth视频")
    print("="*100)
    
    model_results = report["model_results"]
    
    print(f"📁 Ground Truth信息:")
    print(f"   总样本数: {report['ground_truth_info']['total_gt_samples']}")
    print(f"   正例样本: {report['ground_truth_info']['positive_samples']}")
    print(f"   负例样本: {report['ground_truth_info']['negative_samples']}")
    
    print(f"\n📊 各模型实际处理的视频数量:")
    for model_key, metrics in model_results.items():
        note = f" ({metrics.get('note', '')})" if 'note' in metrics else ""
        print(f"   {metrics['model']}: {metrics['sample_size']} 个视频{note}")
    
    print(f"\n🎯 核心性能指标对比:")
    print(f"{'指标':<20} ", end="")
    for model_key, metrics in model_results.items():
        print(f"{metrics['model']:<15} ", end="")
    print()
    print("-" * (20 + 15 * len(model_results)))
    
    metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    for metric in metrics_to_show:
        print(f"{metric:<20} ", end="")
        for model_key, metrics in model_results.items():
            value = metrics[metric]
            print(f"{value:<15.4f} ", end="")
        print()
    
    print(f"\n📈 混淆矩阵对比:")
    for model_key, metrics in model_results.items():
        tp, tn, fp, fn = metrics['true_positives'], metrics['true_negatives'], metrics['false_positives'], metrics['false_negatives']
        print(f"   {metrics['model']}: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    # 找出有效评估的模型（样本数>5）
    valid_models = {k: v for k, v in model_results.items() if v['sample_size'] > 5}
    
    if valid_models:
        print(f"\n💡 主要发现 (基于样本数>5的模型):")
        
        # 找出最佳模型
        best_f1_model = max(valid_models.items(), key=lambda x: x[1]['f1_score'])
        best_precision_model = max(valid_models.items(), key=lambda x: x[1]['precision'])
        best_recall_model = max(valid_models.items(), key=lambda x: x[1]['recall'])
        
        print(f"   🥇 最佳F1分数: {best_f1_model[1]['model']} ({best_f1_model[1]['f1_score']:.4f})")
        print(f"   🎯 最佳精确度: {best_precision_model[1]['model']} ({best_precision_model[1]['precision']:.4f})")
        print(f"   🔍 最佳召回率: {best_recall_model[1]['model']} ({best_recall_model[1]['recall']:.4f})")
    
    print(f"\n⚠️ 重要说明:")
    print("   评估基于各模型实际处理的视频数量")
    print("   GPT-4.1由于Azure内容过滤政策限制，处理样本较少")
    print("   GPT-4o和Gemini提供了更完整的Ground Truth覆盖")
    
    print("="*100)

def main():
    """主函数"""
    print("🚀 实际三模型对比评估")
    print("基于可用的处理结果")
    print("=" * 60)
    
    # 创建对比报告
    report = create_practical_comparison_report()
    
    if report:
        # 打印总结
        print_practical_comparison_summary(report)
        
        print(f"\n🎉 实际对比评估完成！")
        print("   详细结果已保存到 result/comparison/ 目录")
    else:
        print("❌ 实际对比评估失败")

if __name__ == "__main__":
    main()