#!/usr/bin/env python3

"""
GPT-5 失败视频重试脚本
分析中间结果，找出失败的视频并重新处理
"""

import os
import sys
import json
import logging
import datetime
from run_gpt5_ghost_probing_fewshot_dada100 import GPT5GhostProbingRunner

class FailedVideoRetryHandler:
    def __init__(self, intermediate_result_file: str):
        self.intermediate_result_file = intermediate_result_file
        self.setup_logging()
        self.failed_videos = []
        
    def setup_logging(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        log_file = f"retry_failed_videos_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== GPT-5 失败视频重试处理开始 ===")
        
    def analyze_failed_videos(self):
        """分析中间结果，找出失败的视频"""
        if not os.path.exists(self.intermediate_result_file):
            self.logger.error(f"中间结果文件不存在: {self.intermediate_result_file}")
            return False
            
        with open(self.intermediate_result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        detailed_results = results.get("detailed_results", [])
        
        total_videos = len(detailed_results)
        error_count = 0
        success_count = 0
        
        for result in detailed_results:
            video_id = result.get("video_id", "unknown")
            evaluation = result.get("evaluation", "unknown")
            
            if evaluation == "ERROR" or not result.get("raw_result"):
                self.failed_videos.append({
                    "video_id": video_id,
                    "ground_truth": result.get("ground_truth", "unknown"),
                    "reason": "API失败或超时"
                })
                error_count += 1
            else:
                success_count += 1
                
        self.logger.info(f"分析完成 - 总视频: {total_videos}, 成功: {success_count}, 失败: {error_count}")
        self.logger.info(f"需要重试的视频数量: {len(self.failed_videos)}")
        
        if self.failed_videos:
            self.logger.info("失败的视频列表:")
            for i, video in enumerate(self.failed_videos, 1):
                self.logger.info(f"{i}. {video['video_id']} (GT: {video['ground_truth']})")
                
        return True
    
    def create_retry_script(self):
        """创建重试脚本"""
        if not self.failed_videos:
            self.logger.info("没有失败的视频需要重试")
            return
        
        # 准备失败视频数据
        failed_videos_data = []
        for video in self.failed_videos:
            failed_videos_data.append({
                'video_id': video['video_id'],
                'ground_truth_label': video['ground_truth']
            })
            
        script_content = f'''#!/usr/bin/env python3
"""
GPT-5 失败视频重试脚本 - 自动生成
生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
失败视频数量: {len(self.failed_videos)}
"""

import sys
import os
sys.path.append('/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gpt-5')

from run_gpt5_ghost_probing_fewshot_dada100 import GPT5GhostProbingRunner
import pandas as pd
import json

def create_failed_videos_gt():
    """创建失败视频的Ground Truth数据"""
    failed_data = {str(failed_videos_data)}
    
    df = pd.DataFrame(failed_data)
    return df

def main():
    print("=== GPT-5 失败视频重试 ===")
    print(f"需要重试的视频数量: {len(self.failed_videos)}")
    
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
'''
        
        retry_script_path = f"retry_failed_videos_script_{self.timestamp}.py"
        with open(retry_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
            
        self.logger.info(f"重试脚本已创建: {retry_script_path}")
        return retry_script_path
    
    def save_failed_videos_report(self):
        """保存失败视频报告"""
        report = {
            "timestamp": self.timestamp,
            "original_file": self.intermediate_result_file,
            "total_failed": len(self.failed_videos),
            "failed_videos": self.failed_videos,
            "analysis_summary": {
                "main_error_types": ["500 Server Error", "API Timeout", "Connection Error"],
                "recommended_action": "重新运行失败的视频"
            }
        }
        
        report_file = f"failed_videos_report_{self.timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"失败视频报告已保存: {report_file}")
        return report_file

def main():
    # 使用最新的中间结果文件
    intermediate_file = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gpt-5/gpt5_intermediate_30videos_20250808_133908.json"
    
    handler = FailedVideoRetryHandler(intermediate_file)
    
    # 分析失败的视频
    if handler.analyze_failed_videos():
        # 创建重试脚本
        retry_script = handler.create_retry_script()
        
        # 保存报告
        report_file = handler.save_failed_videos_report()
        
        print(f"\\n=== 分析完成 ===")
        print(f"失败视频数量: {len(handler.failed_videos)}")
        print(f"报告文件: {report_file}")
        if retry_script:
            print(f"重试脚本: {retry_script}")
            print(f"运行重试脚本: python {retry_script}")
    
if __name__ == "__main__":
    main()