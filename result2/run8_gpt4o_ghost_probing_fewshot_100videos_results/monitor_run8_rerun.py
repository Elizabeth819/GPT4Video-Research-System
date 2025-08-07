#!/usr/bin/env python3
"""
监控Run 8重新运行进度
"""

import os
import json
import time
import glob

def check_progress():
    """检查Run 8重新运行进度"""
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run8_gpt4o_ghost_probing_fewshot_100videos_results/rerun_corrected"
    
    # 查找中间结果文件
    intermediate_files = glob.glob(os.path.join(output_dir, "run8_rerun_intermediate_*videos_*.json"))
    
    if not intermediate_files:
        print("❌ 未找到中间结果文件")
        return
    
    # 找到最新的中间结果文件
    latest_file = max(intermediate_files, key=os.path.getmtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_count = len(data.get('detailed_results', []))
        total_videos = 100  # 目标处理视频数
        
        print(f"📊 Run 8 重新运行进度报告")
        print(f"🎯 已处理视频: {processed_count}/100")
        print(f"📈 完成度: {processed_count/total_videos*100:.1f}%")
        
        # 统计当前结果
        if processed_count > 0:
            tp_count = sum(1 for r in data['detailed_results'] if r.get('evaluation') == 'TP')
            fp_count = sum(1 for r in data['detailed_results'] if r.get('evaluation') == 'FP')
            tn_count = sum(1 for r in data['detailed_results'] if r.get('evaluation') == 'TN')
            fn_count = sum(1 for r in data['detailed_results'] if r.get('evaluation') == 'FN')
            
            print(f"🔍 当前混淆矩阵:")
            print(f"   True Positives:  {tp_count}")
            print(f"   False Positives: {fp_count}")
            print(f"   True Negatives:  {tn_count}")
            print(f"   False Negatives: {fn_count}")
            
            # 计算当前指标（如果有足够数据）
            if tp_count + fp_count > 0:
                precision = tp_count / (tp_count + fp_count)
                print(f"📏 当前精确度: {precision:.3f} ({precision*100:.1f}%)")
            
            if tp_count + fn_count > 0:
                recall = tp_count / (tp_count + fn_count)
                print(f"🎯 当前召回率: {recall:.3f} ({recall*100:.1f}%)")
            
            if tp_count + fp_count > 0 and tp_count + fn_count > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"🏆 当前F1分数: {f1:.3f} ({f1*100:.1f}%)")
        
        print(f"📁 最新结果文件: {os.path.basename(latest_file)}")
        print(f"⏰ 更新时间: {time.ctime(os.path.getmtime(latest_file))}")
        
    except Exception as e:
        print(f"❌ 读取进度文件失败: {e}")

if __name__ == "__main__":
    check_progress()