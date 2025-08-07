#!/usr/bin/env python3
"""
Monitor Run 19 Progress - Claude 4 Ghost Probing Detection
监控实验进度和中间结果
"""

import json
import os
import glob
import datetime
from collections import Counter

def monitor_run19_progress():
    """监控Run 19进度"""
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run19_claude4"
    
    # 查找最新的中间结果文件
    intermediate_files = glob.glob(os.path.join(output_dir, "run19_intermediate_*videos_*.json"))
    
    if not intermediate_files:
        print("❌ 未找到中间结果文件")
        return
    
    # 获取最新文件
    latest_file = max(intermediate_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        detailed_results = results.get("detailed_results", [])
        total_processed = len(detailed_results)
        
        if total_processed == 0:
            print("📊 实验刚开始，暂无结果")
            return
        
        # 计算性能指标
        evaluations = [r['evaluation'] for r in detailed_results]
        eval_counts = Counter(evaluations)
        
        tp = eval_counts.get('TP', 0)
        fp = eval_counts.get('FP', 0)
        tn = eval_counts.get('TN', 0)
        fn = eval_counts.get('FN', 0)
        errors = eval_counts.get('ERROR', 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        print(f"🎯 Run 19 Claude 4 进度监控")
        print(f"📅 时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 最新文件: {os.path.basename(latest_file)}")
        print(f"📈 已处理: {total_processed}/201 视频 ({total_processed*100/201:.1f}%)")
        print(f"")
        print(f"📊 性能指标:")
        print(f"   精确度: {precision:.3f} ({precision*100:.1f}%)")
        print(f"   召回率: {recall:.3f} ({recall*100:.1f}%)")
        print(f"   F1分数: {f1:.3f} ({f1*100:.1f}%)")
        print(f"   准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"")
        print(f"🔢 混淆矩阵:")
        print(f"   TP (真正例): {tp}")
        print(f"   FP (假正例): {fp}")
        print(f"   TN (真负例): {tn}")
        print(f"   FN (假负例): {fn}")
        print(f"   ERROR (错误): {errors}")
        
        # 显示最近几个结果
        print(f"")
        print(f"🔍 最近5个视频结果:")
        for i, result in enumerate(detailed_results[-5:]):
            video_id = result['video_id']
            gt = result['ground_truth']
            pred = result['key_actions']
            eval_result = result['evaluation']
            print(f"   {total_processed-4+i}: {video_id} | GT: {gt} | 预测: {pred} | 评估: {eval_result}")
        
        # 估算完成时间
        if total_processed > 0:
            # 根据最新的处理速度估算
            file_time = os.path.getctime(latest_file)
            start_time = datetime.datetime.fromtimestamp(file_time - total_processed * 26)  # 估算26秒/视频
            elapsed = datetime.datetime.now() - start_time
            avg_time_per_video = elapsed.total_seconds() / total_processed
            remaining_videos = 201 - total_processed
            estimated_remaining = remaining_videos * avg_time_per_video
            estimated_completion = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining)
            
            print(f"")
            print(f"⏱️ 时间估算:")
            print(f"   平均处理时间: {avg_time_per_video:.1f}秒/视频")
            print(f"   剩余时间: {estimated_remaining/3600:.1f}小时")
            print(f"   预计完成: {estimated_completion.strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ 读取结果文件失败: {str(e)}")

if __name__ == "__main__":
    monitor_run19_progress()