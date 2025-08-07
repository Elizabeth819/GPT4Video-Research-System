#!/usr/bin/env python3
"""
LLaVA Ghost Probing Detection Evaluation Script
评估LLaVA鬼探头检测结果与ground truth的对比分析
计算准确率、精确率、召回率、F1分数等指标
文件路径: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/llava_ghost_probing_evaluation.py
"""

import os
import json
import csv
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/llava_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLaVAGhostProbingEvaluator:
    """LLaVA鬼探头检测评估器"""
    
    def __init__(self, 
                 groundtruth_file: str = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv",
                 output_folder: str = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/evaluation_results"):
        """
        初始化评估器
        
        Args:
            groundtruth_file: Ground truth标签文件路径
            output_folder: 评估结果输出文件夹
        """
        self.groundtruth_file = Path(groundtruth_file)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 加载ground truth
        self.ground_truth = self._load_ground_truth()
        
        logger.info(f"✅ 评估器初始化完成")
        logger.info(f"📋 Ground truth标签: {len(self.ground_truth)}")
        logger.info(f"📁 输出文件夹: {self.output_folder}")
    
    def _load_ground_truth(self) -> Dict[str, str]:
        """加载ground truth标签"""
        try:
            ground_truth = {}
            
            # 读取TSV格式的ground truth文件
            with open(self.groundtruth_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    video_id = row['video_id'].replace('.avi', '')  # 移除扩展名统一格式
                    label = row['ground_truth_label']
                    ground_truth[video_id] = label
            
            logger.info(f"成功加载{len(ground_truth)}个ground truth标签")
            return ground_truth
            
        except Exception as e:
            logger.error(f"加载ground truth失败: {e}")
            return {}
    
    def _standardize_labels(self, gt_label: str, pred_label: str) -> Tuple[str, str]:
        """
        标准化标签格式
        
        Args:
            gt_label: Ground truth标签
            pred_label: 预测标签
            
        Returns:
            (标准化的gt_label, 标准化的pred_label)
        """
        # 标准化ground truth标签
        if gt_label.lower() == 'none':
            gt_standardized = 'normal'
        elif 'ghost probing' in gt_label.lower():
            gt_standardized = 'ghost_probing'
        else:
            gt_standardized = 'normal'
        
        # 标准化预测标签
        if pred_label == 'ghost_probing':
            pred_standardized = 'ghost_probing'
        elif pred_label == 'potential_ghost_probing':
            pred_standardized = 'ghost_probing'  # 将潜在鬼探头归类为鬼探头
        else:
            pred_standardized = 'normal'
        
        return gt_standardized, pred_standardized
    
    def evaluate_results(self, llava_results_file: str) -> Dict:
        """
        评估LLaVA结果
        
        Args:
            llava_results_file: LLaVA结果文件路径
            
        Returns:
            评估指标字典
        """
        logger.info(f"🎯 开始评估LLaVA结果: {llava_results_file}")
        
        try:
            # 加载LLaVA结果
            with open(llava_results_file, 'r', encoding='utf-8') as f:
                llava_data = json.load(f)
            
            # 提取结果数据
            if 'results' in llava_data:
                results = llava_data['results']
            else:
                results = llava_data
            
            # 准备评估数据
            evaluation_data = []
            matched_count = 0
            
            for result in results:
                video_id = result['video_id']
                pred_label = result['ghost_probing_label']
                
                # 查找对应的ground truth
                if video_id in self.ground_truth:
                    gt_label = self.ground_truth[video_id]
                    
                    # 标准化标签
                    gt_std, pred_std = self._standardize_labels(gt_label, pred_label)
                    
                    evaluation_data.append({
                        'video_id': video_id,
                        'ground_truth_raw': gt_label,
                        'prediction_raw': pred_label,
                        'ground_truth': gt_std,
                        'prediction': pred_std,
                        'confidence': result.get('confidence', 0.0),
                        'processing_time': result.get('processing_time', 0.0),
                        'correct': gt_std == pred_std
                    })
                    matched_count += 1
                else:
                    logger.warning(f"视频{video_id}在ground truth中未找到")
            
            logger.info(f"✅ 成功匹配{matched_count}个视频进行评估")
            
            # 计算评估指标
            metrics = self._calculate_metrics(evaluation_data)
            
            # 保存评估结果
            self._save_evaluation_results(evaluation_data, metrics, llava_results_file)
            
            # 生成可视化报告
            self._generate_visualizations(evaluation_data, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {}
    
    def _calculate_metrics(self, evaluation_data: List[Dict]) -> Dict:
        """计算评估指标"""
        try:
            # 提取真实标签和预测标签
            y_true = [item['ground_truth'] for item in evaluation_data]
            y_pred = [item['prediction'] for item in evaluation_data]
            
            # 计算基本指标
            accuracy = accuracy_score(y_true, y_pred)
            
            # 对于二分类（ghost_probing vs normal）
            precision = precision_score(y_true, y_pred, pos_label='ghost_probing', average='binary')
            recall = recall_score(y_true, y_pred, pos_label='ghost_probing', average='binary')
            f1 = f1_score(y_true, y_pred, pos_label='ghost_probing', average='binary')
            
            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred, labels=['normal', 'ghost_probing'])
            
            # 详细分类报告
            class_report = classification_report(y_true, y_pred, output_dict=True)
            
            # 统计信息
            total_videos = len(evaluation_data)
            correct_predictions = sum([item['correct'] for item in evaluation_data])
            
            # 按类别统计
            gt_ghost_count = sum([1 for gt in y_true if gt == 'ghost_probing'])
            gt_normal_count = sum([1 for gt in y_true if gt == 'normal'])
            pred_ghost_count = sum([1 for pred in y_pred if pred == 'ghost_probing'])
            pred_normal_count = sum([1 for pred in y_pred if pred == 'normal'])
            
            # 计算平均置信度
            avg_confidence = np.mean([item['confidence'] for item in evaluation_data])
            avg_processing_time = np.mean([item['processing_time'] for item in evaluation_data])
            
            metrics = {
                'model': 'LLaVA-Video-7B-Qwen2',
                'evaluation_timestamp': self.timestamp,
                'dataset_info': {
                    'total_videos': total_videos,
                    'ground_truth_ghost_probing': gt_ghost_count,
                    'ground_truth_normal': gt_normal_count,
                    'predicted_ghost_probing': pred_ghost_count,
                    'predicted_normal': pred_normal_count
                },
                'performance_metrics': {
                    'accuracy': round(accuracy, 4),
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'f1_score': round(f1, 4),
                    'correct_predictions': correct_predictions,
                    'average_confidence': round(avg_confidence, 4),
                    'average_processing_time': round(avg_processing_time, 2)
                },
                'confusion_matrix': {
                    'matrix': cm.tolist(),
                    'labels': ['normal', 'ghost_probing'],
                    'true_negatives': int(cm[0, 0]),
                    'false_positives': int(cm[0, 1]),
                    'false_negatives': int(cm[1, 0]),
                    'true_positives': int(cm[1, 1])
                },
                'classification_report': class_report,
                'comparison_with_gpt41_balanced': {
                    'gpt41_f1_score': 0.712,
                    'gpt41_recall': 0.963,
                    'gpt41_precision': 0.565,
                    'llava_vs_gpt41_f1_diff': round(f1 - 0.712, 4),
                    'llava_vs_gpt41_recall_diff': round(recall - 0.963, 4),
                    'llava_vs_gpt41_precision_diff': round(precision - 0.565, 4)
                }
            }
            
            logger.info("📊 评估指标计算完成:")
            logger.info(f"  准确率: {accuracy:.4f}")
            logger.info(f"  精确率: {precision:.4f}")
            logger.info(f"  召回率: {recall:.4f}")
            logger.info(f"  F1分数: {f1:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"指标计算失败: {e}")
            return {}
    
    def _save_evaluation_results(self, evaluation_data: List[Dict], metrics: Dict, results_file: str):
        """保存评估结果"""
        try:
            # 保存详细评估数据
            detailed_file = self.output_folder / f"llava_detailed_evaluation_{self.timestamp}.json"
            detailed_results = {
                'source_file': results_file,
                'evaluation_data': evaluation_data,
                'metrics': metrics
            }
            
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            # 保存CSV格式的对比结果
            csv_file = self.output_folder / f"llava_evaluation_comparison_{self.timestamp}.csv"
            df = pd.DataFrame(evaluation_data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            # 保存简化的指标报告
            metrics_file = self.output_folder / f"llava_metrics_summary_{self.timestamp}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 评估结果保存完成:")
            logger.info(f"  详细结果: {detailed_file}")
            logger.info(f"  对比CSV: {csv_file}")
            logger.info(f"  指标总结: {metrics_file}")
            
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")
    
    def _generate_visualizations(self, evaluation_data: List[Dict], metrics: Dict):
        """生成可视化图表"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'LLaVA Ghost Probing Detection Evaluation Results\n{self.timestamp}', fontsize=16)
            
            # 1. 混淆矩阵
            cm = np.array(metrics['confusion_matrix']['matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Ghost Probing'],
                       yticklabels=['Normal', 'Ghost Probing'],
                       ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            
            # 2. 性能指标对比
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            metrics_values = [
                metrics['performance_metrics']['accuracy'],
                metrics['performance_metrics']['precision'],
                metrics['performance_metrics']['recall'],
                metrics['performance_metrics']['f1_score']
            ]
            
            bars = axes[0, 1].bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
            axes[0, 1].set_title('Performance Metrics')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_ylim(0, 1)
            
            # 添加数值标签
            for bar, value in zip(bars, metrics_values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            # 3. 与GPT-4.1的对比
            comparison_data = metrics['comparison_with_gpt41_balanced']
            models = ['LLaVA-Video', 'GPT-4.1 Balanced']
            f1_scores = [metrics['performance_metrics']['f1_score'], comparison_data['gpt41_f1_score']]
            recall_scores = [metrics['performance_metrics']['recall'], comparison_data['gpt41_recall']]
            precision_scores = [metrics['performance_metrics']['precision'], comparison_data['gpt41_precision']]
            
            x = np.arange(len(models))
            width = 0.25
            
            axes[1, 0].bar(x - width, f1_scores, width, label='F1 Score', color='lightblue')
            axes[1, 0].bar(x, recall_scores, width, label='Recall', color='lightgreen')
            axes[1, 0].bar(x + width, precision_scores, width, label='Precision', color='lightcoral')
            
            axes[1, 0].set_title('LLaVA vs GPT-4.1 Balanced Comparison')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(models)
            axes[1, 0].legend()
            axes[1, 0].set_ylim(0, 1)
            
            # 4. 置信度分布
            confidences = [item['confidence'] for item in evaluation_data]
            correct_confidences = [item['confidence'] for item in evaluation_data if item['correct']]
            incorrect_confidences = [item['confidence'] for item in evaluation_data if not item['correct']]
            
            axes[1, 1].hist([correct_confidences, incorrect_confidences], 
                           bins=20, alpha=0.7, label=['Correct', 'Incorrect'],
                           color=['green', 'red'])
            axes[1, 1].set_title('Confidence Distribution')
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = self.output_folder / f"llava_evaluation_visualization_{self.timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 可视化图表保存: {plot_file}")
            
        except Exception as e:
            logger.error(f"生成可视化失败: {e}")
    
    def compare_with_other_models(self, llava_results: str, other_results: Dict[str, str]):
        """
        与其他模型结果进行对比
        
        Args:
            llava_results: LLaVA结果文件路径
            other_results: 其他模型结果文件路径字典 {'model_name': 'result_file_path'}
        """
        logger.info("🔄 开始多模型对比分析")
        
        try:
            # 评估LLaVA
            llava_metrics = self.evaluate_results(llava_results)
            
            # 收集所有模型的指标
            all_metrics = {'LLaVA-Video': llava_metrics}
            
            for model_name, result_file in other_results.items():
                if os.path.exists(result_file):
                    metrics = self.evaluate_results(result_file)
                    all_metrics[model_name] = metrics
                else:
                    logger.warning(f"模型{model_name}的结果文件不存在: {result_file}")
            
            # 生成对比报告
            self._generate_comparison_report(all_metrics)
            
        except Exception as e:
            logger.error(f"多模型对比失败: {e}")
    
    def _generate_comparison_report(self, all_metrics: Dict):
        """生成多模型对比报告"""
        try:
            comparison_report = {
                'timestamp': self.timestamp,
                'models_compared': list(all_metrics.keys()),
                'comparison_metrics': {}
            }
            
            # 提取关键指标进行对比
            for model_name, metrics in all_metrics.items():
                if 'performance_metrics' in metrics:
                    comparison_report['comparison_metrics'][model_name] = {
                        'accuracy': metrics['performance_metrics']['accuracy'],
                        'precision': metrics['performance_metrics']['precision'],
                        'recall': metrics['performance_metrics']['recall'],
                        'f1_score': metrics['performance_metrics']['f1_score']
                    }
            
            # 找出最佳性能
            best_models = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                best_score = 0
                best_model = None
                for model_name, metrics in comparison_report['comparison_metrics'].items():
                    if metrics[metric] > best_score:
                        best_score = metrics[metric]
                        best_model = model_name
                best_models[metric] = {'model': best_model, 'score': best_score}
            
            comparison_report['best_performing'] = best_models
            
            # 保存对比报告
            report_file = self.output_folder / f"multi_model_comparison_{self.timestamp}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 多模型对比报告保存: {report_file}")
            
        except Exception as e:
            logger.error(f"生成对比报告失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLaVA鬼探头检测评估')
    parser.add_argument('--llava-results', required=True,
                       help='LLaVA结果JSON文件路径')
    parser.add_argument('--groundtruth-file',
                       default='/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv',
                       help='Ground truth标签文件路径')
    parser.add_argument('--output-folder',
                       default='/Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/evaluation_results',
                       help='评估结果输出文件夹')
    parser.add_argument('--compare-with', nargs='*',
                       help='其他模型结果文件路径列表')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = LLaVAGhostProbingEvaluator(
        groundtruth_file=args.groundtruth_file,
        output_folder=args.output_folder
    )
    
    # 评估LLaVA结果
    metrics = evaluator.evaluate_results(args.llava_results)
    
    if metrics:
        print("="*60)
        print("LLaVA Ghost Probing Detection Evaluation Results")
        print("="*60)
        print(f"准确率: {metrics['performance_metrics']['accuracy']:.4f}")
        print(f"精确率: {metrics['performance_metrics']['precision']:.4f}")
        print(f"召回率: {metrics['performance_metrics']['recall']:.4f}")
        print(f"F1分数: {metrics['performance_metrics']['f1_score']:.4f}")
        print("="*60)
        
        # 与GPT-4.1对比
        comparison = metrics['comparison_with_gpt41_balanced']
        print("与GPT-4.1 Balanced对比:")
        print(f"F1分数差异: {comparison['llava_vs_gpt41_f1_diff']:+.4f}")
        print(f"召回率差异: {comparison['llava_vs_gpt41_recall_diff']:+.4f}")
        print(f"精确率差异: {comparison['llava_vs_gpt41_precision_diff']:+.4f}")
        print("="*60)
    
    # 多模型对比（如果指定）
    if args.compare_with:
        other_results = {}
        for i, result_file in enumerate(args.compare_with):
            model_name = f"Model_{i+1}"
            other_results[model_name] = result_file
        evaluator.compare_with_other_models(args.llava_results, other_results)

if __name__ == "__main__":
    main()