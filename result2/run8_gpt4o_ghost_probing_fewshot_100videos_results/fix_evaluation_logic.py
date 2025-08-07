#!/usr/bin/env python3
"""
修正Run 8重新运行的评估逻辑错误
"""

import json
import glob
import os
import pandas as pd

def fix_evaluation_logic():
    """修正评估逻辑"""
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8_gpt4o_ghost_probing_fewshot_100videos_results/rerun_corrected"
    
    # 找到最新的中间结果文件
    intermediate_files = glob.glob(os.path.join(output_dir, "run8_rerun_intermediate_*videos_*.json"))
    if not intermediate_files:
        print("❌ 未找到中间结果文件")
        return
    
    latest_file = max(intermediate_files, key=os.path.getmtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 加载ground truth
        gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/labels.csv"
        df = pd.read_csv(gt_path)
        ground_truth_dict = {}
        for index, row in df.iterrows():
            video_id = str(row['video_id']).replace('.avi', '')
            label = str(row['ground_truth_label']).strip()
            if 'ghost probing' in label.lower():
                ground_truth_dict[video_id] = "ghost_probing"
            else:
                ground_truth_dict[video_id] = "none"
        
        # 修正评估
        corrected_results = []
        for result in data.get('detailed_results', []):
            video_id = result.get('video_id', '').replace('.avi', '')
            key_actions = result.get('key_actions', '').lower()
            
            # 正确的预测逻辑
            if 'no ghost probing' in key_actions or 'not ghost probing' in key_actions:
                prediction = "none"
            elif 'ghost probing' in key_actions:
                prediction = "ghost_probing"
            else:
                prediction = "none"
            
            # 获取ground truth
            ground_truth = ground_truth_dict.get(video_id, "unknown")
            
            # 正确的评估
            if ground_truth == "unknown":
                evaluation = "UNKNOWN"
            elif ground_truth == prediction:
                evaluation = "TP" if prediction == "ghost_probing" else "TN"
            else:
                evaluation = "FP" if prediction == "ghost_probing" else "FN"
            
            # 更新结果
            result['evaluation'] = evaluation
            result['prediction'] = prediction
            corrected_results.append(result)
        
        # 统计修正后的结果
        tp = fp = tn = fn = 0
        for result in corrected_results:
            evaluation = result.get('evaluation', '')
            if evaluation == "TP":
                tp += 1
            elif evaluation == "FP":
                fp += 1
            elif evaluation == "TN":
                tn += 1
            elif evaluation == "FN":
                fn += 1
        
        # 计算性能指标
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"🔧 修正后的Run 8重新运行结果 (前{len(corrected_results)}个视频)")
        print("="*60)
        
        print("📋 修正后预测分析:")
        print("视频ID | Ground Truth | 预测 | 评估 | Key Actions")
        print("-" * 65)
        
        ghost_predictions = no_ghost_predictions = 0
        for result in corrected_results:
            video_id = result.get('video_id', '').replace('.avi', '')
            gt = ground_truth_dict.get(video_id, 'unknown')
            evaluation = result.get('evaluation', '')
            prediction = result.get('prediction', '')
            key_actions = result.get('key_actions', '')
            
            if prediction == "ghost_probing":
                ghost_predictions += 1
            else:
                no_ghost_predictions += 1
            
            print(f"{video_id:12} | {gt:12} | {prediction:13} | {evaluation:4} | {key_actions}")
        
        print("\n" + "="*60)
        print("📊 修正后预测分布:")
        print(f"🎯 预测为Ghost Probing: {ghost_predictions}/{len(corrected_results)} ({ghost_predictions/len(corrected_results)*100:.1f}%)")
        print(f"❌ 预测为None: {no_ghost_predictions}/{len(corrected_results)} ({no_ghost_predictions/len(corrected_results)*100:.1f}%)")
        
        print(f"\n🔢 修正后混淆矩阵:")
        print(f"   True Positives:  {tp}")
        print(f"   False Positives: {fp}")
        print(f"   True Negatives:  {tn}")
        print(f"   False Negatives: {fn}")
        
        print(f"\n🏆 修正后性能指标:")
        print(f"   准确率 (Accuracy):    {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   精确度 (Precision):   {precision:.3f} ({precision*100:.1f}%)")
        print(f"   召回率 (Recall):      {recall:.3f} ({recall*100:.1f}%)")
        print(f"   特异性 (Specificity): {specificity:.3f} ({specificity*100:.1f}%)")
        print(f"   平衡准确率:           {balanced_accuracy:.3f} ({balanced_accuracy*100:.1f}%)")
        print(f"   F1分数:              {f1:.3f} ({f1*100:.1f}%)")
        
        print(f"\n📈 对比:")
        print("修正前: F1=85.7%, 召回率=100.0%, 特异性=0.0%")
        print(f"修正后: F1={f1*100:.1f}%, 召回率={recall*100:.1f}%, 特异性={specificity*100:.1f}%")
        
    except Exception as e:
        print(f"❌ 修正失败: {e}")

if __name__ == "__main__":
    fix_evaluation_logic()