#!/usr/bin/env python3
"""
Run 18 Performance Evaluation
评估Gemini-2.0-Flash + Simple Prompt在DADA-100上的性能
"""

import os
import json
import pandas as pd
from pathlib import Path
import datetime

class Run18PerformanceEvaluator:
    def __init__(self):
        self.output_dir = Path(__file__).parent
        self.project_root = Path("/Users/wanmeng/repository/GPT4Video-cobra-auto")
        self.groundtruth_file = self.project_root / "result" / "DADA-100-videos" / "groundtruth_labels.csv"
        
    def load_groundtruth(self):
        """加载ground truth标签"""
        try:
            df = pd.read_csv(self.groundtruth_file, sep='\t')
            groundtruth = {}
            
            for _, row in df.iterrows():
                video_id = row['video_id'].replace('.avi', '')
                label = row['ground_truth_label']
                
                # 处理标签格式
                if pd.isna(label) or label == 'none':
                    groundtruth[video_id] = False
                else:
                    # 包含ghost probing的都标记为True
                    groundtruth[video_id] = 'ghost probing' in str(label).lower()
            
            print(f"✅ Loaded {len(groundtruth)} ground truth labels")
            return groundtruth
            
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return {}
    
    def load_run18_results(self):
        """加载Run 18的分析结果"""
        results = {}
        processed_count = 0
        
        # 遍历所有结果文件
        for result_file in self.output_dir.glob("actionSummary_*.json"):
            try:
                video_id = result_file.stem.replace("actionSummary_", "")
                
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查key_actions字段
                key_actions = data.get('key_actions', '').lower()
                has_ghost_probing = 'ghost probing' in key_actions
                
                results[video_id] = {
                    'predicted': has_ghost_probing,
                    'key_actions': data.get('key_actions', ''),
                    'summary': data.get('summary', ''),
                    'sentiment': data.get('sentiment', ''),
                    'scene_theme': data.get('scene_theme', '')
                }
                processed_count += 1
                
            except Exception as e:
                print(f"⚠️ Error processing {result_file}: {e}")
                continue
        
        print(f"✅ Loaded {processed_count} Run 18 predictions")
        return results
    
    def calculate_metrics(self, groundtruth, predictions):
        """计算性能指标"""
        # 找到共同的视频ID
        common_videos = set(groundtruth.keys()) & set(predictions.keys())
        
        if not common_videos:
            print("❌ No common videos found between ground truth and predictions")
            return None
        
        # 计算混淆矩阵
        tp = fp = tn = fn = 0
        
        detailed_results = []
        
        for video_id in sorted(common_videos):
            gt_label = groundtruth[video_id]
            pred_label = predictions[video_id]['predicted']
            
            if gt_label and pred_label:
                tp += 1
                result_type = "TP"
            elif not gt_label and not pred_label:
                tn += 1
                result_type = "TN"
            elif not gt_label and pred_label:
                fp += 1
                result_type = "FP"
            else:  # gt_label and not pred_label
                fn += 1
                result_type = "FN"
            
            detailed_results.append({
                'video_id': video_id,
                'ground_truth': gt_label,
                'predicted': pred_label,
                'result_type': result_type,
                'key_actions': predictions[video_id]['key_actions'],
                'summary': predictions[video_id]['summary'][:100] + "..."
            })
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        metrics = {
            'total_videos': len(common_videos),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'accuracy': accuracy,
            'detailed_results': detailed_results
        }
        
        return metrics
    
    def print_performance_summary(self, metrics):
        """打印性能摘要"""
        print("\n" + "="*60)
        print("🚀 Run 18 Performance Analysis Summary")
        print("="*60)
        print(f"📋 Model: Gemini-2.0-Flash-exp + Simple Prompt")
        print(f"📊 Dataset: DADA-100 ({metrics['total_videos']} videos)")
        print(f"📅 Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "-"*40)
        print("📈 Performance Metrics:")
        print("-"*40)
        print(f"🎯 F1-Score:    {metrics['f1_score']:.3f}")
        print(f"🔍 Precision:   {metrics['precision']:.3f}")
        print(f"📡 Recall:      {metrics['recall']:.3f}")
        print(f"🛡️  Specificity: {metrics['specificity']:.3f}")
        print(f"✅ Accuracy:    {metrics['accuracy']:.3f}")
        print("\n" + "-"*40)
        print("🧮 Confusion Matrix:")
        print("-"*40)
        print(f"True Positives (TP):  {metrics['tp']}")
        print(f"False Positives (FP): {metrics['fp']}")
        print(f"True Negatives (TN):  {metrics['tn']}")
        print(f"False Negatives (FN): {metrics['fn']}")
        
        # 错误案例分析
        print("\n" + "-"*40)
        print("❌ Error Analysis:")
        print("-"*40)
        
        fp_cases = [r for r in metrics['detailed_results'] if r['result_type'] == 'FP']
        fn_cases = [r for r in metrics['detailed_results'] if r['result_type'] == 'FN']
        
        print(f"False Positives ({len(fp_cases)} cases):")
        for case in fp_cases[:5]:  # 显示前5个
            print(f"  • {case['video_id']}: {case['key_actions']} - {case['summary']}")
        
        print(f"\nFalse Negatives ({len(fn_cases)} cases):")
        for case in fn_cases[:5]:  # 显示前5个
            print(f"  • {case['video_id']}: {case['key_actions']} - {case['summary']}")
    
    def save_detailed_results(self, metrics):
        """保存详细评估结果"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细结果
        detailed_file = self.output_dir / f"run18_detailed_evaluation_{timestamp}.json"
        evaluation_data = {
            'run_info': {
                'run_id': 'Run 18',
                'model': 'gemini-2.0-flash-exp',
                'prompt_type': 'Simple Paper Batch (No Few-shot)',
                'evaluation_timestamp': timestamp
            },
            'performance_metrics': {
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'specificity': metrics['specificity'],
                'accuracy': metrics['accuracy'],
                'total_videos': metrics['total_videos']
            },
            'confusion_matrix': {
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'tn': metrics['tn'],
                'fn': metrics['fn']
            },
            'detailed_results': metrics['detailed_results']
        }
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {detailed_file}")
        
        # 保存CSV格式的详细结果
        csv_file = self.output_dir / f"run18_detailed_results_{timestamp}.csv"
        df = pd.DataFrame(metrics['detailed_results'])
        df.to_csv(csv_file, index=False)
        print(f"📋 CSV results saved to: {csv_file}")
        
        return evaluation_data
    
    def run_evaluation(self):
        """运行完整评估"""
        print("🚀 Starting Run 18 Performance Evaluation...")
        
        # 加载数据
        groundtruth = self.load_groundtruth()
        predictions = self.load_run18_results()
        
        if not groundtruth or not predictions:
            print("❌ Failed to load data")
            return None
        
        # 计算指标
        metrics = self.calculate_metrics(groundtruth, predictions)
        
        if not metrics:
            print("❌ Failed to calculate metrics")
            return None
        
        # 显示结果
        self.print_performance_summary(metrics)
        
        # 保存结果
        evaluation_data = self.save_detailed_results(metrics)
        
        return evaluation_data

def main():
    """主函数"""
    evaluator = Run18PerformanceEvaluator()
    result = evaluator.run_evaluation()
    
    if result:
        print("\n✅ Run 18 evaluation completed successfully!")
        print(f"🎯 F1-Score: {result['performance_metrics']['f1_score']:.3f}")
    else:
        print("\n❌ Run 18 evaluation failed!")

if __name__ == "__main__":
    main()