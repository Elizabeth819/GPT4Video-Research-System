#!/usr/bin/env python3
"""
Azure ML A100 GPU Ghost Probing Batch Processing Script
使用平衡版GPT-4.1 prompt处理images_1_001到images_5_XXX的100个视频进行鬼探头打标
输出格式与groundtruth.txt保持一致，便于准确率、精确度、召回率等指标对比
"""

import os
import sys
import json
import csv
import logging
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional

# 导入现有的ActionSummary模块
sys.path.append('/Users/wanmeng/repository/GPT4Video-cobra-auto')
from ActionSummary import process_video_with_retry, get_video_duration, log_execution_time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ghost_probing_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GhostProbingBatchProcessor:
    def __init__(self, 
                 video_folder: str = "DADA-2000-videos",
                 output_folder: str = "result/ghost_probing_gpt41_balanced",
                 groundtruth_file: str = "result/groundtruth_labels.csv"):
        """
        初始化Ghost Probing批处理器
        
        Args:
            video_folder: 视频文件夹路径
            output_folder: 输出文件夹路径
            groundtruth_file: Ground truth标签文件路径
        """
        self.video_folder = Path(video_folder)
        self.output_folder = Path(output_folder)
        self.groundtruth_file = Path(groundtruth_file)
        
        # 创建输出文件夹
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # 时间戳用于文件命名
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 加载ground truth数据
        self.ground_truth = self._load_ground_truth()
        
        # 获取目标视频列表
        self.target_videos = self._get_target_videos()
        
        logger.info(f"初始化完成: 找到{len(self.target_videos)}个目标视频")
    
    def _load_ground_truth(self) -> Dict[str, str]:
        """加载ground truth标签数据"""
        try:
            df = pd.read_csv(self.groundtruth_file, sep='\t')
            ground_truth = {}
            for _, row in df.iterrows():
                video_id = row['video_id']
                label = row['ground_truth_label']
                ground_truth[video_id] = label
            logger.info(f"加载ground truth数据: {len(ground_truth)}个标签")
            return ground_truth
        except Exception as e:
            logger.error(f"无法加载ground truth文件: {e}")
            return {}
    
    def _get_target_videos(self) -> List[str]:
        """获取目标视频列表 (images_1_001到images_5_XXX的前100个)"""
        target_videos = []
        
        # 获取所有符合条件的视频
        for i in range(1, 6):  # images_1_* 到 images_5_*
            pattern = f"images_{i}_*.avi"
            videos = sorted(self.video_folder.glob(pattern))
            for video in videos:
                if len(target_videos) >= 100:
                    break
                target_videos.append(video.name)
            if len(target_videos) >= 100:
                break
        
        # 确保只取前100个
        target_videos = target_videos[:100]
        logger.info(f"目标视频列表: {len(target_videos)}个视频")
        return target_videos
    
    def _extract_ghost_probing_from_json(self, json_result: Dict) -> Tuple[str, Optional[str]]:
        """
        从JSON结果中提取鬼探头信息
        
        Args:
            json_result: 视频分析JSON结果
            
        Returns:
            Tuple[detection_result, timestamp]: (检测结果, 时间戳)
        """
        try:
            # 检查key_actions字段
            key_actions = json_result.get('key_actions', '').lower()
            
            # 查找鬼探头关键词
            if 'ghost probing' in key_actions:
                # 尝试提取时间戳
                timestamp = self._extract_timestamp_from_result(json_result)
                return "ghost probing", timestamp
            elif 'potential ghost probing' in key_actions:
                timestamp = self._extract_timestamp_from_result(json_result)
                return "potential ghost probing", timestamp
            else:
                return "none", None
                
        except Exception as e:
            logger.error(f"提取鬼探头信息失败: {e}")
            return "none", None
    
    def _extract_timestamp_from_result(self, json_result: Dict) -> Optional[str]:
        """从JSON结果中提取时间戳"""
        try:
            # 尝试从Start_Timestamp提取
            start_timestamp = json_result.get('Start_Timestamp', '')
            if start_timestamp:
                # 提取数字部分
                import re
                match = re.search(r'(\d+\.?\d*)', start_timestamp)
                if match:
                    return f"{int(float(match.group(1)))}s"
            
            # 备选方案：从其他字段提取
            segment_id = json_result.get('segment_id', '')
            if segment_id:
                match = re.search(r'(\d+)', segment_id)
                if match:
                    return f"{match.group(1)}s"
            
            return None
        except Exception as e:
            logger.error(f"提取时间戳失败: {e}")
            return None
    
    def _format_result_for_comparison(self, video_id: str, detection_result: str, timestamp: Optional[str] = None) -> str:
        """
        格式化结果以匹配groundtruth.txt格式
        
        Args:
            video_id: 视频ID
            detection_result: 检测结果
            timestamp: 时间戳（如果有）
            
        Returns:
            格式化的结果字符串
        """
        if detection_result == "none":
            return "none"
        elif timestamp:
            return f"{timestamp}: {detection_result}"
        else:
            return detection_result
    
    @log_execution_time
    def process_single_video(self, video_name: str) -> Dict:
        """
        处理单个视频
        
        Args:
            video_name: 视频文件名
            
        Returns:
            处理结果字典
        """
        video_path = self.video_folder / video_name
        
        try:
            logger.info(f"开始处理视频: {video_name}")
            
            # 使用现有的process_video_with_retry函数
            result = process_video_with_retry(
                str(video_path),
                frame_interval=10,  # 每10秒一个段落
                frames_per_interval=10,  # 每段落10帧
                skip_audio=False,
                use_balanced_prompt=True  # 使用平衡版prompt
            )
            
            # 提取鬼探头信息
            detection_result = "none"
            timestamp = None
            
            if result and 'segments' in result:
                for segment in result['segments']:
                    ghost_result, ghost_timestamp = self._extract_ghost_probing_from_json(segment)
                    if ghost_result != "none":
                        detection_result = ghost_result
                        timestamp = ghost_timestamp
                        break  # 找到第一个就停止
            
            # 格式化结果
            formatted_result = self._format_result_for_comparison(video_name, detection_result, timestamp)
            
            # 获取ground truth
            ground_truth_label = self.ground_truth.get(video_name, "unknown")
            
            result_dict = {
                'video_id': video_name,
                'predicted_label': formatted_result,
                'ground_truth_label': ground_truth_label,
                'detection_result': detection_result,
                'timestamp': timestamp,
                'processing_status': 'success'
            }
            
            logger.info(f"视频 {video_name} 处理完成: {formatted_result}")
            return result_dict
            
        except Exception as e:
            logger.error(f"处理视频 {video_name} 失败: {e}")
            return {
                'video_id': video_name,
                'predicted_label': 'processing_failed',
                'ground_truth_label': self.ground_truth.get(video_name, "unknown"),
                'detection_result': 'error',
                'timestamp': None,
                'processing_status': 'failed',
                'error_message': str(e)
            }
    
    def process_batch(self, max_videos: int = 100) -> List[Dict]:
        """
        批处理所有目标视频
        
        Args:
            max_videos: 最大处理视频数量
            
        Returns:
            处理结果列表
        """
        results = []
        processed_count = 0
        
        logger.info(f"开始批处理 {min(len(self.target_videos), max_videos)} 个视频")
        
        for video_name in self.target_videos[:max_videos]:
            try:
                result = self.process_single_video(video_name)
                results.append(result)
                processed_count += 1
                
                # 每处理10个视频保存一次中间结果
                if processed_count % 10 == 0:
                    self._save_intermediate_results(results)
                    logger.info(f"已处理 {processed_count}/{min(len(self.target_videos), max_videos)} 个视频")
                    
            except Exception as e:
                logger.error(f"处理视频 {video_name} 时发生错误: {e}")
                results.append({
                    'video_id': video_name,
                    'predicted_label': 'processing_failed',
                    'ground_truth_label': self.ground_truth.get(video_name, "unknown"),
                    'detection_result': 'error',
                    'timestamp': None,
                    'processing_status': 'failed',
                    'error_message': str(e)
                })
                processed_count += 1
        
        logger.info(f"批处理完成: 总共处理 {processed_count} 个视频")
        return results
    
    def _save_intermediate_results(self, results: List[Dict]):
        """保存中间结果"""
        try:
            intermediate_file = self.output_folder / f"intermediate_results_{self.timestamp}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存中间结果失败: {e}")
    
    def save_results(self, results: List[Dict]):
        """
        保存最终结果
        
        Args:
            results: 处理结果列表
        """
        try:
            # 保存详细JSON结果
            json_file = self.output_folder / f"ghost_probing_results_{self.timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存CSV格式用于对比
            csv_file = self.output_folder / f"ghost_probing_comparison_{self.timestamp}.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['video_id', 'predicted_label', 'ground_truth_label', 'processing_status'])
                
                for result in results:
                    writer.writerow([
                        result['video_id'],
                        result['predicted_label'],
                        result['ground_truth_label'],
                        result['processing_status']
                    ])
            
            logger.info(f"结果已保存: {json_file} 和 {csv_file}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        计算准确率、精确度、召回率等指标
        
        Args:
            results: 处理结果列表
            
        Returns:
            指标字典
        """
        try:
            # 统计各种情况
            tp = 0  # True Positive: 预测有鬼探头，实际也有
            fp = 0  # False Positive: 预测有鬼探头，实际没有
            tn = 0  # True Negative: 预测没有鬼探头，实际也没有
            fn = 0  # False Negative: 预测没有鬼探头，实际有
            
            successful_results = [r for r in results if r['processing_status'] == 'success']
            
            for result in successful_results:
                predicted = result['predicted_label']
                ground_truth = result['ground_truth_label']
                
                # 判断是否为鬼探头
                predicted_ghost = 'ghost probing' in predicted.lower() if predicted != 'none' else False
                ground_truth_ghost = 'ghost probing' in ground_truth.lower() if ground_truth != 'none' else False
                
                if predicted_ghost and ground_truth_ghost:
                    tp += 1
                elif predicted_ghost and not ground_truth_ghost:
                    fp += 1
                elif not predicted_ghost and ground_truth_ghost:
                    fn += 1
                else:
                    tn += 1
            
            # 计算指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'total_videos': len(results),
                'successful_processing': len(successful_results),
                'failed_processing': len(results) - len(successful_results),
                'true_positive': tp,
                'false_positive': fp,
                'true_negative': tn,
                'false_negative': fn,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"计算指标失败: {e}")
            return {}
    
    def generate_report(self, results: List[Dict], metrics: Dict):
        """
        生成处理报告
        
        Args:
            results: 处理结果列表
            metrics: 计算得到的指标
        """
        try:
            report_file = self.output_folder / f"ghost_probing_report_{self.timestamp}.md"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# Ghost Probing Detection Report\n\n")
                f.write(f"## 处理概况\n")
                f.write(f"- 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- 总视频数: {metrics.get('total_videos', 0)}\n")
                f.write(f"- 成功处理: {metrics.get('successful_processing', 0)}\n")
                f.write(f"- 处理失败: {metrics.get('failed_processing', 0)}\n\n")
                
                f.write("## 性能指标\n")
                f.write(f"- **准确率 (Accuracy)**: {metrics.get('accuracy', 0):.3f}\n")
                f.write(f"- **精确度 (Precision)**: {metrics.get('precision', 0):.3f}\n")
                f.write(f"- **召回率 (Recall)**: {metrics.get('recall', 0):.3f}\n")
                f.write(f"- **F1分数**: {metrics.get('f1_score', 0):.3f}\n")
                f.write(f"- **误报率**: {metrics.get('false_positive_rate', 0):.3f}\n\n")
                
                f.write("## 混淆矩阵\n")
                f.write(f"- True Positive (TP): {metrics.get('true_positive', 0)}\n")
                f.write(f"- False Positive (FP): {metrics.get('false_positive', 0)}\n")
                f.write(f"- True Negative (TN): {metrics.get('true_negative', 0)}\n")
                f.write(f"- False Negative (FN): {metrics.get('false_negative', 0)}\n\n")
                
                # 详细结果分析
                f.write("## 详细结果分析\n\n")
                f.write("### 检测到的鬼探头案例\n")
                ghost_cases = [r for r in results if 'ghost probing' in r.get('predicted_label', '').lower()]
                for case in ghost_cases:
                    f.write(f"- {case['video_id']}: {case['predicted_label']} (GT: {case['ground_truth_label']})\n")
                
                f.write("\n### 漏检案例\n")
                missed_cases = [r for r in results if 'ghost probing' in r.get('ground_truth_label', '').lower() and 'ghost probing' not in r.get('predicted_label', '').lower()]
                for case in missed_cases:
                    f.write(f"- {case['video_id']}: {case['predicted_label']} (GT: {case['ground_truth_label']})\n")
            
            logger.info(f"报告已生成: {report_file}")
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Ghost Probing Batch Processing with Balanced GPT-4.1')
    parser.add_argument('--video-folder', default='DADA-2000-videos', help='视频文件夹路径')
    parser.add_argument('--output-folder', default='result/ghost_probing_gpt41_balanced', help='输出文件夹路径')
    parser.add_argument('--groundtruth-file', default='result/groundtruth_labels.csv', help='Ground truth文件路径')
    parser.add_argument('--max-videos', type=int, default=100, help='最大处理视频数量')
    parser.add_argument('--dry-run', action='store_true', help='仅预览不实际处理')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = GhostProbingBatchProcessor(
        video_folder=args.video_folder,
        output_folder=args.output_folder,
        groundtruth_file=args.groundtruth_file
    )
    
    if args.dry_run:
        logger.info("=== 预览模式 ===")
        logger.info(f"将处理以下 {min(len(processor.target_videos), args.max_videos)} 个视频:")
        for i, video in enumerate(processor.target_videos[:args.max_videos]):
            gt_label = processor.ground_truth.get(video, "unknown")
            logger.info(f"{i+1:3d}. {video} (GT: {gt_label})")
        return
    
    # 开始批处理
    logger.info("=== 开始批处理 ===")
    results = processor.process_batch(max_videos=args.max_videos)
    
    # 计算指标
    logger.info("=== 计算指标 ===")
    metrics = processor.calculate_metrics(results)
    
    # 保存结果
    logger.info("=== 保存结果 ===")
    processor.save_results(results)
    
    # 生成报告
    logger.info("=== 生成报告 ===")
    processor.generate_report(results, metrics)
    
    # 输出总结
    logger.info("=== 处理完成 ===")
    logger.info(f"总共处理: {len(results)} 个视频")
    logger.info(f"成功处理: {metrics.get('successful_processing', 0)} 个")
    logger.info(f"处理失败: {metrics.get('failed_processing', 0)} 个")
    logger.info(f"准确率: {metrics.get('accuracy', 0):.3f}")
    logger.info(f"精确度: {metrics.get('precision', 0):.3f}")
    logger.info(f"召回率: {metrics.get('recall', 0):.3f}")
    logger.info(f"F1分数: {metrics.get('f1_score', 0):.3f}")


if __name__ == "__main__":
    main()