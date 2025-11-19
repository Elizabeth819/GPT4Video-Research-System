#!/usr/bin/env python3
"""
GPT-5 失败视频重试脚本 - 自动生成
生成时间: 2025-08-08 14:51:04
失败视频数量: 23
"""

import sys
import os
sys.path.append('/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gpt-5')

from run_gpt5_ghost_probing_fewshot_dada100 import GPT5GhostProbingRunner
import pandas as pd
import json

def create_failed_videos_gt():
    """创建失败视频的Ground Truth数据"""
    failed_data = [{'video_id': 'images_1_001.avi', 'ground_truth_label': 'none'}, {'video_id': 'images_1_002.avi', 'ground_truth_label': '5s: ghost probing'}, {'video_id': 'images_1_003.avi', 'ground_truth_label': '2s: ghost probing'}, {'video_id': 'images_1_004.avi', 'ground_truth_label': 'none'}, {'video_id': 'images_1_005.avi', 'ground_truth_label': '8s: ghost probing'}, {'video_id': 'images_1_006.avi', 'ground_truth_label': '9s: ghost probing'}, {'video_id': 'images_1_007.avi', 'ground_truth_label': '6s: ghost probing'}, {'video_id': 'images_1_008.avi', 'ground_truth_label': '3s: ghost probing'}, {'video_id': 'images_1_009.avi', 'ground_truth_label': 'none'}, {'video_id': 'images_1_011.avi', 'ground_truth_label': '11s: ghost probing'}, {'video_id': 'images_1_012.avi', 'ground_truth_label': '11s: ghost probing'}, {'video_id': 'images_1_016.avi', 'ground_truth_label': '4s: ghost probing'}, {'video_id': 'images_1_017.avi', 'ground_truth_label': '17s: ghost probing'}, {'video_id': 'images_1_018.avi', 'ground_truth_label': 'none'}, {'video_id': 'images_1_019.avi', 'ground_truth_label': 'none'}, {'video_id': 'images_1_022.avi', 'ground_truth_label': '5s: ghost probing'}, {'video_id': 'images_1_023.avi', 'ground_truth_label': 'none'}, {'video_id': 'images_1_024.avi', 'ground_truth_label': 'none'}, {'video_id': 'images_1_025.avi', 'ground_truth_label': 'none'}, {'video_id': 'images_1_026.avi', 'ground_truth_label': 'none'}, {'video_id': 'images_2_001.avi', 'ground_truth_label': 'none'}, {'video_id': 'images_2_002.avi', 'ground_truth_label': 'none'}, {'video_id': 'images_2_003.avi', 'ground_truth_label': 'none'}]
    
    df = pd.DataFrame(failed_data)
    return df

def main():
    print("=== GPT-5 失败视频重试 ===")
    print(f"需要重试的视频数量: 23")
    
    # 创建特殊的Runner实例，只处理失败的视频
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gpt-5"
    complex_prompt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/VIP/prompts/paper_batch_original_complex_prompt.txt"
    fewshot_prompt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/VIP/prompts/few-shot prompt/run8_gpt4o_fewshot_examples.txt"
    
    # 创建临时的runner实例
    runner = GPT5GhostProbingRunner(
        output_dir=output_dir + "/retry",
        complex_prompt_path=complex_prompt_path,
        fewshot_prompt_path=fewshot_prompt_path
    )
    
    # 替换ground truth数据为失败的视频
    runner.ground_truth = create_failed_videos_gt()
    
    print("开始重新处理失败的视频...")
    runner.run()
    print("重试完成!")

if __name__ == "__main__":
    main()
