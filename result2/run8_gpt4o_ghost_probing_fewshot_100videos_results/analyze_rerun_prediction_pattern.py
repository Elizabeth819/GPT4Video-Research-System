#!/usr/bin/env python3
"""
分析Run 8重新运行的预测模式，特别关注特异性和是否倾向于预测全部正例
"""

import json
import glob
import os

def analyze_prediction_pattern():
    """分析预测模式"""
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
        
        results = data.get('detailed_results', [])
        if not results:
            print("❌ 没有详细结果")
            return
        
        print(f"🔍 Run 8 重新运行预测模式分析 (前{len(results)}个视频)")
        print("="*60)
        
        # 统计混淆矩阵
        tp = fp = tn = fn = 0
        ghost_predictions = no_ghost_predictions = 0
        ghost_truth = no_ghost_truth = 0
        
        # 详细分析每个预测
        print("\n📋 详细预测分析:")
        print("视频ID | Ground Truth | 预测 | 评估 | Key Actions")
        print("-" * 65)
        
        for result in results:
            video_id = result.get('video_id', '').replace('.avi', '')
            gt = result.get('ground_truth', '')
            evaluation = result.get('evaluation', '')
            key_actions = result.get('key_actions', '')
            
            # 判断预测
            if 'ghost probing' in key_actions.lower():
                prediction = "ghost_probing"
                ghost_predictions += 1
            else:
                prediction = "none"
                no_ghost_predictions += 1
            
            # 判断ground truth
            if gt == "ghost_probing":
                ghost_truth += 1
            else:
                no_ghost_truth += 1
            
            # 统计混淆矩阵
            if evaluation == "TP":
                tp += 1
            elif evaluation == "FP":
                fp += 1
            elif evaluation == "TN":
                tn += 1
            elif evaluation == "FN":
                fn += 1
            
            print(f"{video_id:12} | {gt:12} | {prediction:13} | {evaluation:4} | {key_actions}")
        
        print("\n" + "="*60)
        print("📊 预测分布统计:")
        print(f"🎯 预测为Ghost Probing: {ghost_predictions}/{len(results)} ({ghost_predictions/len(results)*100:.1f}%)")
        print(f"❌ 预测为None: {no_ghost_predictions}/{len(results)} ({no_ghost_predictions/len(results)*100:.1f}%)")
        
        print(f"\n🏷️ Ground Truth分布:")
        print(f"🎯 实际Ghost Probing: {ghost_truth}/{len(results)} ({ghost_truth/len(results)*100:.1f}%)")
        print(f"❌ 实际None: {no_ghost_truth}/{len(results)} ({no_ghost_truth/len(results)*100:.1f}%)")
        
        print(f"\n🔢 混淆矩阵:")
        print(f"   True Positives:  {tp}")
        print(f"   False Positives: {fp}")
        print(f"   True Negatives:  {tn}")
        print(f"   False Negatives: {fn}")
        
        # 计算性能指标
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n🏆 性能指标:")
        print(f"   准确率 (Accuracy):    {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   精确度 (Precision):   {precision:.3f} ({precision*100:.1f}%)")
        print(f"   召回率 (Recall):      {recall:.3f} ({recall*100:.1f}%)")
        print(f"   特异性 (Specificity): {specificity:.3f} ({specificity*100:.1f}%)")
        print(f"   平衡准确率:           {balanced_accuracy:.3f} ({balanced_accuracy*100:.1f}%)")
        print(f"   F1分数:              {f1:.3f} ({f1*100:.1f}%)")
        
        print(f"\n⚠️ 关键观察:")
        if tn == 0:
            print("🚨 特异性为0%！模型没有正确识别任何负样本！")
            print("🚨 这表明模型倾向于将所有或大部分视频预测为ghost probing")
        elif specificity < 0.5:
            print(f"⚠️ 特异性较低 ({specificity*100:.1f}%)，模型容易产生误报")
        
        if fp > tp:
            print(f"🚨 误报数量 ({fp}) 超过真正例 ({tp})！")
        
        prediction_bias = ghost_predictions / len(results)
        if prediction_bias > 0.8:
            print(f"🚨 强烈的正例偏向：{prediction_bias*100:.1f}%的视频被预测为ghost probing")
        elif prediction_bias > 0.6:
            print(f"⚠️ 明显的正例偏向：{prediction_bias*100:.1f}%的视频被预测为ghost probing")
        
        print(f"\n📈 与原始Run 8对比预期:")
        print(f"原始Run 8 (校正后): F1=65.0%, 召回率=80.0%, 特异性=32.7%")
        print(f"当前重新运行:       F1={f1*100:.1f}%, 召回率={recall*100:.1f}%, 特异性={specificity*100:.1f}%")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

if __name__ == "__main__":
    analyze_prediction_pattern()