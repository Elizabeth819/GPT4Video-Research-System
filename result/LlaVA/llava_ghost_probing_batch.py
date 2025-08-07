#!/usr/bin/env python3
"""
LLaVA-NeXT Ghost Probing Batch Processing Script
批量处理100个DADA视频进行鬼探头检测
使用LLaVA-Video-7B-Qwen2模型和平衡提示词
文件路径: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/llava_ghost_probing_batch.py
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
from tqdm import tqdm
import time

# 导入LLaVA鬼探头检测器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llava_ghost_probing_detector import LLaVAGhostProbingDetector

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llava_ghost_probing_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLaVAGhostProbingBatchProcessor:
    """LLaVA鬼探头批处理器"""
    
    def __init__(self, 
                 video_folder: str = "./inputs/video_data",
                 output_folder: str = "./outputs/results", 
                 groundtruth_file: str = "groundtruth_labels.csv"):
        """
        初始化LLaVA鬼探头批处理器
        
        Args:
            video_folder: DADA-100视频文件夹路径
            output_folder: 输出结果文件夹路径
            groundtruth_file: Ground truth标签文件路径
        """
        self.video_folder = Path(video_folder)
        self.output_folder = Path(output_folder)
        self.groundtruth_file = Path(groundtruth_file)
        
        # 创建输出文件夹
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # 时间戳用于文件命名
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 初始化LLaVA检测器
        logger.info("正在初始化LLaVA鬼探头检测器...")
        self.detector = LLaVAGhostProbingDetector()
        
        # 获取目标视频列表
        self.target_videos = self._get_target_videos()
        
        # 加载ground truth数据（如果存在）
        self.ground_truth = self._load_ground_truth()
        
        logger.info(f"✅ 批处理器初始化完成")
        logger.info(f"📁 视频文件夹: {self.video_folder}")
        logger.info(f"📁 输出文件夹: {self.output_folder}")
        logger.info(f"🎬 目标视频数量: {len(self.target_videos)}")
        logger.info(f"📋 Ground truth标签: {len(self.ground_truth)}")
    
    def _get_target_videos(self) -> List[str]:
        """获取目标视频列表 (DADA-100视频)"""
        target_videos = []
        
        try:
            # 获取所有符合DADA命名规范的视频
            for video_file in sorted(self.video_folder.glob("images_*.avi")):
                if video_file.is_file():
                    target_videos.append(video_file.name)
            
            # 限制为前100个视频
            target_videos = target_videos[:100]
            
            logger.info(f"找到{len(target_videos)}个目标视频")
            return target_videos
            
        except Exception as e:
            logger.error(f"获取目标视频列表失败: {e}")
            return []
    
    def _load_ground_truth(self) -> Dict[str, str]:
        """加载ground truth标签数据"""
        try:
            if not self.groundtruth_file.exists():
                logger.warning(f"Ground truth文件不存在: {self.groundtruth_file}")
                return {}
            
            # 尝试不同的分隔符和编码
            separators = ['\t', ',', ';']
            encodings = ['utf-8', 'gbk', 'gb2312']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        df = pd.read_csv(self.groundtruth_file, sep=sep, encoding=encoding)
                        if 'video_id' in df.columns and len(df) > 0:
                            ground_truth = {}
                            for _, row in df.iterrows():
                                video_id = row['video_id']
                                # 尝试不同的标签列名
                                for col in ['ground_truth_label', 'label', 'ghost_probing', 'gt_label']:
                                    if col in row:
                                        ground_truth[video_id] = row[col]
                                        break
                            logger.info(f"成功加载ground truth: {len(ground_truth)}个标签 (编码: {encoding}, 分隔符: '{sep}')")
                            return ground_truth
                    except Exception:
                        continue
            
            logger.warning("无法解析ground truth文件")
            return {}
            
        except Exception as e:
            logger.error(f"加载ground truth失败: {e}")
            return {}
    
    def process_single_video(self, video_filename: str) -> Optional[Dict]:
        """
        处理单个视频
        
        Args:
            video_filename: 视频文件名
            
        Returns:
            处理结果字典，失败返回None
        """
        try:
            video_path = self.video_folder / video_filename
            video_id = video_filename.replace('.avi', '')
            
            logger.info(f"🎬 处理视频: {video_id}")
            
            # 使用LLaVA进行分析
            start_time = time.time()
            analysis_result = self.detector.analyze_video(str(video_path), video_id)
            processing_time = time.time() - start_time
            
            if analysis_result:
                # 提取鬼探头标签
                ghost_label, confidence = self.detector.extract_ghost_probing_label(analysis_result)
                
                # 构建结果
                result = {
                    'video_id': video_id,
                    'filename': video_filename,
                    'processing_time': round(processing_time, 2),
                    'ghost_probing_label': ghost_label,
                    'confidence': confidence,
                    'llava_analysis': analysis_result,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 添加ground truth信息（如果有）
                if video_id in self.ground_truth:
                    result['ground_truth'] = self.ground_truth[video_id]
                
                logger.info(f"✅ 视频处理成功: {video_id} -> {ghost_label} (置信度: {confidence})")
                return result
            else:
                logger.error(f"❌ 视频分析失败: {video_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 处理视频失败 {video_filename}: {e}")
            return None
    
    def process_batch(self, 
                     start_index: int = 0, 
                     limit: Optional[int] = None,
                     save_interval: int = 10) -> Dict:
        """
        批量处理视频
        
        Args:
            start_index: 开始索引
            limit: 处理数量限制
            save_interval: 保存间隔
            
        Returns:
            批处理统计结果
        """
        logger.info(f"🚀 开始批量处理视频")
        logger.info(f"📊 开始索引: {start_index}")
        logger.info(f"📊 处理限制: {limit if limit else '无限制'}")
        
        # 确定处理范围
        videos_to_process = self.target_videos[start_index:]
        if limit:
            videos_to_process = videos_to_process[:limit]
        
        logger.info(f"📊 本次将处理 {len(videos_to_process)} 个视频")
        
        # 统计信息
        stats = {
            'total_videos': len(videos_to_process),
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'ghost_probing_detected': 0,
            'potential_ghost_probing_detected': 0,
            'normal_detected': 0,
            'start_time': datetime.now().isoformat(),
            'results': []
        }
        
        # 批处理主循环
        with tqdm(total=len(videos_to_process), desc="处理视频") as pbar:
            for i, video_filename in enumerate(videos_to_process):
                try:
                    # 处理单个视频
                    result = self.process_single_video(video_filename)
                    
                    stats['processed'] += 1
                    
                    if result:
                        stats['successful'] += 1
                        stats['results'].append(result)
                        
                        # 统计检测结果
                        label = result['ghost_probing_label']
                        if label == 'ghost_probing':
                            stats['ghost_probing_detected'] += 1
                        elif label == 'potential_ghost_probing':
                            stats['potential_ghost_probing_detected'] += 1
                        else:
                            stats['normal_detected'] += 1
                    else:
                        stats['failed'] += 1
                    
                    # 定期保存结果
                    if (i + 1) % save_interval == 0:
                        self._save_intermediate_results(stats, i + 1)
                    
                    # 更新进度条
                    pbar.set_postfix({
                        '成功': stats['successful'],
                        '失败': stats['failed'],
                        '鬼探头': stats['ghost_probing_detected']
                    })
                    pbar.update(1)
                    
                except KeyboardInterrupt:
                    logger.warning("⚠️ 用户中断处理")
                    break
                except Exception as e:
                    logger.error(f"❌ 批处理异常: {e}")
                    stats['failed'] += 1
                    pbar.update(1)
        
        # 完成统计
        stats['end_time'] = datetime.now().isoformat()
        stats['success_rate'] = stats['successful'] / stats['processed'] if stats['processed'] > 0 else 0
        
        # 保存最终结果
        self._save_final_results(stats)
        
        # 输出总结
        self._print_summary(stats)
        
        return stats
    
    def _save_intermediate_results(self, stats: Dict, current_index: int):
        """保存中间结果"""
        try:
            intermediate_file = self.output_folder / f"llava_ghost_probing_intermediate_{current_index}_{self.timestamp}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 保存中间结果: {intermediate_file}")
        except Exception as e:
            logger.error(f"保存中间结果失败: {e}")
    
    def _save_final_results(self, stats: Dict):
        """保存最终结果"""
        try:
            # 保存完整JSON结果
            json_file = self.output_folder / f"llava_ghost_probing_final_{self.timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            # 保存CSV格式结果（便于对比分析）
            csv_file = self.output_folder / f"llava_ghost_probing_results_{self.timestamp}.csv"
            self._save_csv_results(stats['results'], csv_file)
            
            # 保存简化版本（与GPT-4.1格式对齐）
            simplified_file = self.output_folder / f"llava_ghost_probing_simplified_{self.timestamp}.json"
            self._save_simplified_results(stats['results'], simplified_file)
            
            logger.info(f"💾 最终结果保存完成:")
            logger.info(f"📄 JSON格式: {json_file}")
            logger.info(f"📊 CSV格式: {csv_file}")
            logger.info(f"📋 简化格式: {simplified_file}")
            
        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")
    
    def _save_csv_results(self, results: List[Dict], csv_file: Path):
        """保存CSV格式结果"""
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                if results:
                    fieldnames = ['video_id', 'ghost_probing_label', 'confidence', 'processing_time', 'ground_truth']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for result in results:
                        row = {
                            'video_id': result['video_id'],
                            'ghost_probing_label': result['ghost_probing_label'],
                            'confidence': result['confidence'],
                            'processing_time': result['processing_time'],
                            'ground_truth': result.get('ground_truth', '')
                        }
                        writer.writerow(row)
        except Exception as e:
            logger.error(f"保存CSV结果失败: {e}")
    
    def _save_simplified_results(self, results: List[Dict], simplified_file: Path):
        """保存简化格式结果（与其他模型对比）"""
        try:
            simplified = {
                'model': 'LLaVA-Video-7B-Qwen2',
                'timestamp': self.timestamp,
                'total_videos': len(results),
                'results': {}
            }
            
            for result in results:
                video_id = result['video_id']
                simplified['results'][video_id] = {
                    'ghost_probing_detection': result['ghost_probing_label'],
                    'confidence': result['confidence'],
                    'key_actions': result['llava_analysis'].get('key_actions', ''),
                    'summary': result['llava_analysis'].get('summary', '')
                }
            
            with open(simplified_file, 'w', encoding='utf-8') as f:
                json.dump(simplified, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存简化结果失败: {e}")
    
    def _print_summary(self, stats: Dict):
        """打印处理总结"""
        logger.info("=" * 60)
        logger.info("🎯 LLaVA鬼探头检测批处理完成")
        logger.info("=" * 60)
        logger.info(f"📊 总计处理: {stats['processed']} 个视频")
        logger.info(f"✅ 成功处理: {stats['successful']} 个")
        logger.info(f"❌ 处理失败: {stats['failed']} 个")
        logger.info(f"📈 成功率: {stats['success_rate']:.1%}")
        logger.info("-" * 40)
        logger.info(f"🚨 鬼探头检测: {stats['ghost_probing_detected']} 个")
        logger.info(f"⚠️  潜在鬼探头: {stats['potential_ghost_probing_detected']} 个")
        logger.info(f"✔️  正常情况: {stats['normal_detected']} 个")
        logger.info("-" * 40)
        logger.info(f"⏰ 开始时间: {stats['start_time']}")
        logger.info(f"⏰ 结束时间: {stats['end_time']}")
        logger.info("=" * 60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLaVA鬼探头检测批处理')
    parser.add_argument('--video-folder', 
                       default='./inputs/video_data',
                       help='视频文件夹路径')
    parser.add_argument('--output-folder',
                       default='./outputs/results',
                       help='输出文件夹路径')
    parser.add_argument('--start-index', type=int, default=0,
                       help='开始处理的视频索引')
    parser.add_argument('--limit', type=int, default=None,
                       help='处理视频数量限制')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='保存中间结果的间隔')
    
    args = parser.parse_args()
    
    # 创建批处理器
    processor = LLaVAGhostProbingBatchProcessor(
        video_folder=args.video_folder,
        output_folder=args.output_folder
    )
    
    # 开始批处理
    stats = processor.process_batch(
        start_index=args.start_index,
        limit=args.limit,
        save_interval=args.save_interval
    )
    
    return stats

if __name__ == "__main__":
    main()