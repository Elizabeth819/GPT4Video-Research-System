#!/usr/bin/env python3
"""
重新计算Run 8的准确率等指标 - 使用最新的labels.csv文件
Run 8: GPT-4o + Paper Batch + Few-shot Learning
"""

import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import datetime

class Run8MetricsCorrectedCalculator:
    def __init__(self):
        self.project_root = Path("/Users/wanmeng/repository/GPT4Video-cobra-auto")
        self.run8_dir = Path(__file__).parent
        # 使用正确的labels.csv文件
        self.ground_truth_file = self.project_root / "result" / "DADA-100-videos" / "labels.csv"
        
    def load_ground_truth_corrected(self):
        """使用最新的labels.csv文件加载ground truth标签"""
        try:
            print(f"📖 Loading ground truth from: {self.ground_truth_file}")
            
            ground_truth = {}
            
            # 读取CSV文件
            df = pd.read_csv(self.ground_truth_file)
            
            for index, row in df.iterrows():
                video_id = str(row['video_id']).replace('.avi', '')
                label = str(row['ground_truth_label']).strip()
                
                # 判断是否包含ghost probing (忽略cut-in，只考虑ghost probing)
                if 'ghost probing' in label.lower():
                    ground_truth[video_id] = 1  # ghost probing
                else:
                    ground_truth[video_id] = 0  # none (包括cut-in也算作none)
            
            print(f"✅ Loaded ground truth for {len(ground_truth)} videos")
            
            # 统计分布
            ghost_count = sum(ground_truth.values())
            none_count = len(ground_truth) - ghost_count
            print(f"📊 Ground Truth Distribution:")
            print(f"   Ghost Probing: {ghost_count}")
            print(f"   None/Cut-in: {none_count}")
            
            return ground_truth
            
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return {}
    
    def load_predictions(self):
        """加载Run 8的预测结果"""
        predictions = {}
        
        # Run 8使用不同的结构，读取最终结果文件
        result_file = self.run8_dir / "run8_ghost_probing_100videos_results" / "run8_final_results_20250727_093406.json"
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 遍历详细结果
            for result in data.get('detailed_results', []):
                video_id = result.get('video_id', '').replace('.avi', '')
                key_actions = result.get('key_actions', '').lower()
                
                # 判断是否预测为ghost probing
                if 'ghost probing' in key_actions:
                    predictions[video_id] = 1
                elif 'no ghost probing' in key_actions or 'not' in key_actions:
                    predictions[video_id] = 0
                else:
                    # 如果不明确，默认为无ghost probing
                    predictions[video_id] = 0
                    
        except Exception as e:
            print(f"❌ Error loading Run 8 results: {e}")
            return {}
        
        print(f"✅ Loaded predictions for {len(predictions)} videos")
        return predictions
    
    def calculate_metrics(self):
        """计算性能指标"""
        print("🚀 Starting Run 8 Performance Evaluation (Corrected)...")
        
        # 加载数据
        ground_truth = self.load_ground_truth_corrected()
        predictions = self.load_predictions()
        
        if not ground_truth or not predictions:
            print("❌ Failed to load data")
            return
        
        # 找到共同的视频
        common_videos = set(ground_truth.keys()) & set(predictions.keys())
        print(f"📊 Common videos for evaluation: {len(common_videos)}")
        
        if len(common_videos) == 0:
            print("❌ No common videos found")
            return
        
        # 准备数据
        y_true = []
        y_pred = []
        detailed_results = []
        
        for video_id in sorted(common_videos):
            gt = ground_truth[video_id]
            pred = predictions[video_id]
            
            y_true.append(gt)
            y_pred.append(pred)
            
            detailed_results.append({
                "video_id": video_id,
                "ground_truth": "ghost_probing" if gt == 1 else "none",
                "prediction": "ghost_probing" if pred == 1 else "none",
                "correct": gt == pred
            })
        
        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 计算特异性和平衡准确率
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        
        # 统计分布
        gt_ghost_count = sum(y_true)
        gt_none_count = len(y_true) - gt_ghost_count
        pred_ghost_count = sum(y_pred)
        pred_none_count = len(y_pred) - pred_ghost_count
        
        # 打印结果
        print("\n" + "="*60)
        print("🎯 Run 8: GPT-4o + Paper Batch + Few-shot Performance Metrics (CORRECTED)")
        print("="*60)
        print(f"📊 Dataset: {len(common_videos)} videos")
        print(f"🎯 Ghost Probing Ground Truth: {gt_ghost_count}")
        print(f"📄 None Ground Truth: {gt_none_count}")
        print()
        print("🏆 PERFORMANCE METRICS:")
        print(f"   Accuracy:     {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Precision:    {precision:.3f} ({precision*100:.1f}%)")
        print(f"   Recall:       {recall:.3f} ({recall*100:.1f}%)")
        print(f"   F1-Score:     {f1:.3f} ({f1*100:.1f}%)")
        print(f"   Specificity:  {specificity:.3f} ({specificity*100:.1f}%)")
        print(f"   Balanced Acc: {balanced_accuracy:.3f} ({balanced_accuracy*100:.1f}%)")
        print()
        print("📈 CONFUSION MATRIX:")
        print(f"   True Positives:  {tp}")
        print(f"   True Negatives:  {tn}")
        print(f"   False Positives: {fp}")
        print(f"   False Negatives: {fn}")
        print()
        print("🔍 PREDICTION DISTRIBUTION:")
        print(f"   Predicted Ghost Probing: {pred_ghost_count}")
        print(f"   Predicted None: {pred_none_count}")
        print("="*60)
        
        # 保存结果
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.run8_dir / f"run8_performance_metrics_corrected_{timestamp}.json"
        
        results = {
            "experiment_info": {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": "GPT-4o",
                "experiment": "Run 8 - GPT-4o + Paper Batch + Few-shot - CORRECTED LABELS",
                "ground_truth_file": str(self.ground_truth_file),
                "total_videos": len(common_videos),
                "ground_truth_distribution": {
                    "ghost_probing": gt_ghost_count,
                    "none": gt_none_count
                },
                "prediction_distribution": {
                    "ghost_probing": pred_ghost_count,
                    "none": pred_none_count
                }
            },
            "performance_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy
            },
            "confusion_matrix": {
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn)
            },
            "detailed_results": detailed_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Metrics saved to: {output_file}")
        print("\n✅ Run 8 evaluation completed successfully!")
        print("🎯 GPT-4o + Paper Batch + Few-shot performance evaluated with CORRECTED labels!")

if __name__ == "__main__":
    calculator = Run8MetricsCorrectedCalculator()
    calculator.calculate_metrics()