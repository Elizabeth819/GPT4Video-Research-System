#!/usr/bin/env python3
"""
仅处理失败视频的简化重试脚本
直接基于原始runner，只处理明确失败的23个视频
"""

import pandas as pd
import os
import sys

# 确保能import原始runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_gpt5_ghost_probing_fewshot_dada100 import GPT5GhostProbingRunner

def main():
    # 失败的23个视频清单（从分析结果中获取）
    failed_videos_data = [
        {'video_id': 'images_1_001.avi', 'ground_truth_label': 'none'},
        {'video_id': 'images_1_002.avi', 'ground_truth_label': '5s: ghost probing'},
        {'video_id': 'images_1_003.avi', 'ground_truth_label': '2s: ghost probing'},
        {'video_id': 'images_1_004.avi', 'ground_truth_label': 'none'},
        {'video_id': 'images_1_005.avi', 'ground_truth_label': '8s: ghost probing'},
        {'video_id': 'images_1_006.avi', 'ground_truth_label': '9s: ghost probing'},
        {'video_id': 'images_1_007.avi', 'ground_truth_label': '6s: ghost probing'},
        {'video_id': 'images_1_008.avi', 'ground_truth_label': '3s: ghost probing'},
        {'video_id': 'images_1_009.avi', 'ground_truth_label': 'none'},
        {'video_id': 'images_1_011.avi', 'ground_truth_label': '11s: ghost probing'},
        {'video_id': 'images_1_012.avi', 'ground_truth_label': '11s: ghost probing'},
        {'video_id': 'images_1_016.avi', 'ground_truth_label': '4s: ghost probing'},
        {'video_id': 'images_1_017.avi', 'ground_truth_label': '17s: ghost probing'},
        {'video_id': 'images_1_018.avi', 'ground_truth_label': 'none'},
        {'video_id': 'images_1_019.avi', 'ground_truth_label': 'none'},
        {'video_id': 'images_1_022.avi', 'ground_truth_label': '5s: ghost probing'},
        {'video_id': 'images_1_023.avi', 'ground_truth_label': 'none'},
        {'video_id': 'images_1_024.avi', 'ground_truth_label': 'none'},
        {'video_id': 'images_1_025.avi', 'ground_truth_label': 'none'},
        {'video_id': 'images_1_026.avi', 'ground_truth_label': 'none'},
        {'video_id': 'images_2_001.avi', 'ground_truth_label': 'none'},
        {'video_id': 'images_2_002.avi', 'ground_truth_label': 'none'},
        {'video_id': 'images_2_003.avi', 'ground_truth_label': 'none'}
    ]
    
    print(f"=== GPT-5 失败视频重试 (简化版) ===")
    print(f"处理失败视频数量: {len(failed_videos_data)}")
    
    # 创建基本配置
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gpt-5/retry"
    complex_prompt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/VIP/prompts/paper_batch_original_complex_prompt.txt"
    fewshot_prompt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/VIP/prompts/few-shot prompt/run8_gpt4o_fewshot_examples.txt"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 创建runner实例
        runner = GPT5GhostProbingRunner(
            output_dir=output_dir,
            complex_prompt_path=complex_prompt_path,
            fewshot_prompt_path=fewshot_prompt_path
        )
        
        # 替换ground truth为失败的视频
        runner.ground_truth = pd.DataFrame(failed_videos_data)
        
        print("开始重新处理失败的视频...")
        runner.run()
        print("重试完成!")
        
    except Exception as e:
        print(f"重试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()