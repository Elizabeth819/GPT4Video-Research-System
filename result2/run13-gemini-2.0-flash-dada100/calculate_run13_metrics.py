#!/usr/bin/env python3
"""
计算Run 13的准确率、精确度、召回率、F1分数等指标
"""

import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import datetime

class Run13MetricsCalculator:
    def __init__(self):
        self.project_root = Path("/Users/wanmeng/repository/GPT4Video-cobra-auto")
        self.run13_dir = Path(__file__).parent
        self.ground_truth_file = self.project_root / "result" / "DADA-100-videos" / "groundtruth_labels.csv"
        
    def load_ground_truth(self):
        """加载ground truth标签"""
        try:
            gt_df = pd.read_csv(self.ground_truth_file, sep='\t')
            
            # 将ground truth转换为二进制标签
            ground_truth = {}
            for _, row in gt_df.iterrows():
                video_id = row['video_id'].replace('.avi', '')  # 移除.avi后缀
                label = row['ground_truth_label']
                
                # 判断是否包含ghost probing
                if 'ghost probing' in str(label).lower():
                    ground_truth[video_id] = 1  # ghost probing
                else:
                    ground_truth[video_id] = 0  # none
            
            print(f"✅ Loaded ground truth for {len(ground_truth)} videos")
            return ground_truth
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return None
    
    def load_run13_predictions(self):
        """加载Run 13的预测结果"""
        try:
            predictions = {}
            result_files = list(self.run13_dir.glob("actionSummary_images_*.json"))
            
            for result_file in result_files:
                video_id = result_file.stem.replace("actionSummary_", "")
                
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    
                    key_actions = result.get('key_actions', '').lower()
                    
                    # 判断预测结果
                    if 'ghost probing' in key_actions:
                        predictions[video_id] = 1  # ghost probing
                    else:
                        predictions[video_id] = 0  # none
                        
                except Exception as e:
                    print(f"⚠️  Error processing {result_file}: {e}")
                    continue
            
            print(f"✅ Loaded predictions for {len(predictions)} videos")
            return predictions
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            return None
    
    def calculate_metrics(self, ground_truth, predictions):
        """计算性能指标"""
        try:
            # 获取共同的视频ID
            common_videos = set(ground_truth.keys()) & set(predictions.keys())
            print(f"📊 Common videos for evaluation: {len(common_videos)}")
            
            if len(common_videos) == 0:
                print("❌ No common videos found!")
                return None
            
            # 构建标签数组
            y_true = []
            y_pred = []
            video_details = []
            
            for video_id in sorted(common_videos):
                gt_label = ground_truth[video_id]
                pred_label = predictions[video_id]
                
                y_true.append(gt_label)
                y_pred.append(pred_label)
                
                video_details.append({
                    'video_id': video_id,
                    'ground_truth': 'ghost_probing' if gt_label == 1 else 'none',
                    'prediction': 'ghost_probing' if pred_label == 1 else 'none',
                    'correct': gt_label == pred_label
                })
            
            # 计算指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # 计算额外指标
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_accuracy = (recall + specificity) / 2
            
            # 统计ground truth分布
            gt_positive = sum(y_true)
            gt_negative = len(y_true) - gt_positive
            
            # 统计预测分布
            pred_positive = sum(y_pred)
            pred_negative = len(y_pred) - pred_positive
            
            metrics = {
                'experiment_info': {
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model': 'Gemini 2.0 Flash',
                    'experiment': 'Run 13 - VIP Prompt',
                    'total_videos': len(common_videos),
                    'ground_truth_distribution': {
                        'ghost_probing': int(gt_positive),
                        'none': int(gt_negative)
                    },
                    'prediction_distribution': {
                        'ghost_probing': int(pred_positive),
                        'none': int(pred_negative)
                    }
                },
                'performance_metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'specificity': float(specificity),
                    'balanced_accuracy': float(balanced_accuracy)
                },
                'confusion_matrix': {
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn)
                },
                'detailed_results': video_details
            }
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return None
    
    def print_metrics_summary(self, metrics):
        """打印指标摘要"""
        if not metrics:
            return
            
        print("\n" + "="*60)
        print("🎯 Run 13: Gemini 2.0 Flash Performance Metrics")
        print("="*60)
        
        exp_info = metrics['experiment_info']
        perf_metrics = metrics['performance_metrics']
        cm = metrics['confusion_matrix']
        
        print(f"📊 Dataset: {exp_info['total_videos']} videos")
        print(f"🎯 Ghost Probing Ground Truth: {exp_info['ground_truth_distribution']['ghost_probing']}")
        print(f"📄 None Ground Truth: {exp_info['ground_truth_distribution']['none']}")
        print()
        
        print("🏆 PERFORMANCE METRICS:")
        print(f"   Accuracy:     {perf_metrics['accuracy']:.3f} ({perf_metrics['accuracy']*100:.1f}%)")
        print(f"   Precision:    {perf_metrics['precision']:.3f} ({perf_metrics['precision']*100:.1f}%)")
        print(f"   Recall:       {perf_metrics['recall']:.3f} ({perf_metrics['recall']*100:.1f}%)")
        print(f"   F1-Score:     {perf_metrics['f1_score']:.3f} ({perf_metrics['f1_score']*100:.1f}%)")
        print(f"   Specificity:  {perf_metrics['specificity']:.3f} ({perf_metrics['specificity']*100:.1f}%)")
        print(f"   Balanced Acc: {perf_metrics['balanced_accuracy']:.3f} ({perf_metrics['balanced_accuracy']*100:.1f}%)")
        print()
        
        print("📈 CONFUSION MATRIX:")
        print(f"   True Positives:  {cm['true_positives']}")
        print(f"   True Negatives:  {cm['true_negatives']}")
        print(f"   False Positives: {cm['false_positives']}")
        print(f"   False Negatives: {cm['false_negatives']}")
        print()
        
        print("🔍 PREDICTION DISTRIBUTION:")
        print(f"   Predicted Ghost Probing: {exp_info['prediction_distribution']['ghost_probing']}")
        print(f"   Predicted None: {exp_info['prediction_distribution']['none']}")
        print("="*60)
    
    def save_metrics(self, metrics):
        """保存指标到文件"""
        if not metrics:
            return
            
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = self.run13_dir / f"run13_performance_metrics_{timestamp}.json"
        
        try:
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Metrics saved to: {metrics_file}")
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")
    
    def run_evaluation(self):
        """运行完整评估"""
        print("🚀 Starting Run 13 Performance Evaluation...")
        
        # 加载数据
        ground_truth = self.load_ground_truth()
        if ground_truth is None:
            return
        
        predictions = self.load_run13_predictions()
        if predictions is None:
            return
        
        # 计算指标
        metrics = self.calculate_metrics(ground_truth, predictions)
        if metrics is None:
            return
        
        # 显示结果
        self.print_metrics_summary(metrics)
        
        # 保存结果
        self.save_metrics(metrics)
        
        return metrics

def main():
    """主函数"""
    calculator = Run13MetricsCalculator()
    metrics = calculator.run_evaluation()
    
    if metrics:
        print("\n✅ Evaluation completed successfully!")
    else:
        print("\n❌ Evaluation failed!")

if __name__ == "__main__":
    main()